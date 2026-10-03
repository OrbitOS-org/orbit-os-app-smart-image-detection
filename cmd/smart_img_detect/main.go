// Edge AI – Smart Image Detection: an Orbit OS app that runs YOLOv8 object detection
// on images you upload, using the on-device Orbit OS AI service (TFLite).
//
// Embedded in the binary: metadata.json and everything under static/ (HTML, CSS, logo, favicon).
// Not embedded: the models/ folder in the working directory — see modelPath.
package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	aiv26 "github.com/OrbitOS-org/orbit-os-sdk-go/v26/api/ai_service/v26"
	orbitos "github.com/OrbitOS-org/orbit-os-sdk-go/v26/client"
	"github.com/OrbitOS-org/orbit-os-sdk-go/v26/logger"
	"github.com/OrbitOS-org/orbit-os-sdk-go/v26/metadata"
)

//go:embed metadata.json
var identityJSON []byte

//go:embed static/*
var staticFS embed.FS

var (
	indexHTML    []byte
	appCSS       []byte
	orbitLogoPNG []byte
	faviconSVG   []byte
)

func init() {
	must := func(name string) []byte {
		b, err := staticFS.ReadFile(name)
		if err != nil {
			panic("smart_img_detect: embed " + name + ": " + err.Error())
		}
		return b
	}
	indexHTML = must("static/index.html")
	appCSS = must("static/app.css")
	orbitLogoPNG = must("static/orbit-logo.png")
	faviconSVG = must("static/favicon.svg")
}

var appManifest = metadata.MustParseAppManifestJSON(identityJSON)

const logTag = "main"

const maxUploadBytes = 32 << 20 // 32 MiB

// Fixed configuration.
const (
	modelID = "yolo8-spotlab"
	// models/ is resolved from the working directory: on the device that is the app data
	// folder (the .orb ships data/models/yolov8_spotlab.tflite).
	modelPath      = "models/yolov8_spotlab.tflite"
	httpListen     = "127.0.0.1:9005"
	execMode       = "cpu" // cpu | gpu | threads
	confThreshold  = float32(0.40)
	iouThreshold   = float32(0.45)
	minLogit       = float32(0.5)
	inferDeadline  = 120 * time.Second
	inputDtypeMode = "auto" // auto | float32 | uint8 — see overrideDtype
)

type inferResponse struct {
	OK          bool            `json:"ok"`
	Error       string          `json:"error,omitempty"`
	RTTMs       float64         `json:"rtt_ms,omitempty"`
	LatencyUs   int64           `json:"latency_us,omitempty"`
	Detections  []DetectionJSON `json:"detections,omitempty"`
	Count       int             `json:"count,omitempty"`
	OutputShape []int32         `json:"output_shape,omitempty"`
}

func main() {
	// -host is only used off-device (development from a laptop over TCP + mTLS).
	// On the device the SDK connects through the local Unix socket.
	host := flag.String("host", "192.168.1.100", "Device IP address (development only)")
	flag.Parse()

	meta := metadata.Build(appManifest)
	logger.Init(appManifest.PackageId, "INFO", true)
	logger.Infof(logTag, "starting %s v%s on http://%s", meta.Name, meta.Version, httpListen)

	exec := aiv26.ExecutionMode_EXEC_CPU
	switch execMode {
	case "gpu":
		exec = aiv26.ExecutionMode_EXEC_GPU
	case "threads":
		exec = aiv26.ExecutionMode_EXEC_HIGH_THREADS
	}

	client, err := orbitos.NewClientAuto(*host)
	if err != nil {
		logger.Fatalf(logTag, "connect: %v", err)
		os.Exit(1)
	}
	defer client.Close()

	ai := client.AIManager

	model, err := uploadAndLoadModel(ai, exec)
	if err != nil {
		logger.Fatalf(logTag, "model: %v", err)
		os.Exit(1)
	}

	var inputTI *aiv26.TensorInfo
	if inputs := model.Response.GetInputs(); len(inputs) > 0 {
		inputTI = inputs[0]
	}
	inputTI = overrideDtype(inputTI, inputDtypeMode)

	srv := &http.Server{
		Addr: httpListen,
		Handler: newHandler(&handlerState{
			ai:           ai,
			model:        model,
			modelID:      modelID,
			modelPath:    modelPath,
			exec:         exec,
			inputTI:      inputTI,
			conf:         confThreshold,
			iou:          iouThreshold,
			minLogit:     minLogit,
			inferTimeout: inferDeadline,
		}),
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 120 * time.Second,
	}

	go func() {
		logger.Infof(logTag, "ready — open http://%s", httpListen)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf(logTag, "http: %v", err)
			os.Exit(1)
		}
	}()

	if client.AppHubManager != nil {
		go func() {
			if err := client.AppHubManager.RegisterWebUI(httpListen, "/edge-ai"); err != nil {
				logger.Warnf(logTag, "AppHub RegisterWebUI: %v", err)
			}
		}()
	}

	// Graceful shutdown on SIGINT / SIGTERM so the TCP socket is released immediately.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	logger.Infof(logTag, "signal %s received — shutting down...", sig)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		logger.Warnf(logTag, "http shutdown: %v", err)
	}
	logger.Infof(logTag, "stopped")
}

type handlerState struct {
	mu           sync.Mutex
	ai           *orbitos.AIManager
	model        *orbitos.AIModel
	modelID      string
	modelPath    string
	exec         aiv26.ExecutionMode
	inputTI      *aiv26.TensorInfo
	conf         float32
	iou          float32
	minLogit     float32
	inferTimeout time.Duration
}

// uploadAndLoadModel uploads the model to the AI service and loads it; returns the handle
// used for inference. The AI service skips the transfer when the model is already loaded.
func uploadAndLoadModel(ai *orbitos.AIManager, exec aiv26.ExecutionMode) (*orbitos.AIModel, error) {
	if _, statErr := os.Stat(modelPath); statErr != nil {
		return nil, fmt.Errorf("model file not found at %q (cwd=%s): %w — place models/ at the app root",
			modelPath, mustGetwd(), statErr)
	}
	logger.Infof(logTag, "uploading model id=%q from %q", modelID, modelPath)
	return ai.UploadAndLoadModel(modelID, modelPath, aiv26.ModelBackend_TFLITE, exec)
}

func newHandler(st *handlerState) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(indexHTML)
	})
	mux.HandleFunc("GET /app.css", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/css; charset=utf-8")
		w.Header().Set("Cache-Control", "public, max-age=3600")
		_, _ = w.Write(appCSS)
	})
	mux.HandleFunc("GET /orbit-logo.png", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Cache-Control", "public, max-age=86400")
		_, _ = w.Write(orbitLogoPNG)
	})
	serveFavicon := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/svg+xml; charset=utf-8")
		w.Header().Set("Cache-Control", "public, max-age=86400")
		_, _ = w.Write(faviconSVG)
	}
	mux.HandleFunc("GET /favicon.svg", serveFavicon)
	// Browsers often request /favicon.ico by default; serve same SVG so the tab icon loads.
	mux.HandleFunc("GET /favicon.ico", serveFavicon)
	mux.HandleFunc("POST /api/infer", func(w http.ResponseWriter, r *http.Request) {
		handleInfer(w, r, st)
	})
	return mux
}

func inferErrNeedsModelReload(err error) bool {
	for e := err; e != nil; e = errors.Unwrap(e) {
		s := strings.ToLower(e.Error())
		if strings.Contains(s, "not loaded") || strings.Contains(s, "no tensor schema") {
			return true
		}
		if strings.Contains(s, "not found") && strings.Contains(s, "model") {
			return true
		}
	}
	return false
}

// ensureModelLoadedAfterInferFailure reloads the model after Infer failed with an error
// that indicates the device lost the model or tensor schema.
//
// We intentionally do not call IsModelLoaded for an early return: after an Orbit runtime
// restart the service can still report stale loaded=true with non-empty inputs while Infer
// fails (e.g. "not loaded" on the backend). Trusting IsModelLoaded here would skip
// re-upload and the retry would never recover.
//
// Caller must hold st.mu.
func (st *handlerState) ensureModelLoadedAfterInferFailure() error {
	if _, statErr := os.Stat(st.modelPath); statErr != nil {
		return fmt.Errorf("model file not found at %q (cwd=%s): %w", st.modelPath, mustGetwd(), statErr)
	}
	if err := st.model.Unload(); err != nil {
		logger.Infof(logTag, "UnloadModel %q before reload (best-effort): %v", st.modelID, err)
	}
	logger.Infof(logTag, "re-uploading model id=%q from %q (after infer failure)", st.modelID, st.modelPath)
	model, err := st.ai.UploadAndLoadModel(st.modelID, st.modelPath, aiv26.ModelBackend_TFLITE, st.exec)
	if err != nil {
		return err
	}
	st.model = model
	inputs := model.Response.GetInputs()
	var ti *aiv26.TensorInfo
	if len(inputs) > 0 {
		ti = inputs[0]
	}
	st.inputTI = overrideDtype(ti, inputDtypeMode)
	return nil
}

func handleInfer(w http.ResponseWriter, r *http.Request, st *handlerState) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		writeJSON(w, http.StatusBadRequest, inferResponse{OK: false, Error: fmt.Sprintf("multipart: %v", err)})
		return
	}
	fh, _, err := r.FormFile("image")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, inferResponse{OK: false, Error: "missing or invalid form field \"image\""})
		return
	}
	defer fh.Close()

	imgBytes, err := io.ReadAll(fh)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, inferResponse{OK: false, Error: fmt.Sprintf("read image: %v", err)})
		return
	}

	runInfer := func() (*aiv26.InferResponse, time.Duration, error) {
		st.mu.Lock()
		model, inputTI := st.model, st.inputTI
		st.mu.Unlock()
		tensor, err := buildImageInputTensorFromReader(bytes.NewReader(imgBytes), inputTI, true)
		if err != nil {
			return nil, 0, err
		}
		ctx, cancel := context.WithTimeout(r.Context(), st.inferTimeout)
		defer cancel()
		t0 := time.Now()
		resp, err := model.Infer(ctx, tensor.Data, tensor.Shape, tensor.Dtype)
		rtt := time.Since(t0)
		return resp, rtt, err
	}

	resp, rtt, err := runInfer()
	if err != nil && inferErrNeedsModelReload(err) {
		st.mu.Lock()
		reloadErr := st.ensureModelLoadedAfterInferFailure()
		st.mu.Unlock()
		if reloadErr != nil {
			writeJSON(w, http.StatusOK, inferResponse{OK: false, Error: fmt.Sprintf("reload model: %v", reloadErr)})
			return
		}
		resp, rtt, err = runInfer()
	}
	if err != nil {
		writeJSON(w, http.StatusOK, inferResponse{OK: false, Error: err.Error()})
		return
	}

	outShape := resp.GetOutputShape()
	outData := bytesToFloat32(resp.GetOutputData())
	dets := detectionsToJSON(outShape, outData, st.conf, st.iou, st.minLogit)

	writeJSON(w, http.StatusOK, inferResponse{
		OK:          true,
		RTTMs:       float64(rtt.Nanoseconds()) / 1e6,
		LatencyUs:   resp.GetLatencyUs(),
		Detections:  dets,
		Count:       len(dets),
		OutputShape: outShape,
	})
}

func mustGetwd() string {
	wd, err := os.Getwd()
	if err != nil {
		return "?"
	}
	return wd
}

func writeJSON(w http.ResponseWriter, status int, v inferResponse) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func overrideDtype(ti *aiv26.TensorInfo, flag string) *aiv26.TensorInfo {
	var want aiv26.TensorDataType
	switch flag {
	case "uint8":
		want = aiv26.TensorDataType_TENSOR_UINT8
	case "float32":
		want = aiv26.TensorDataType_TENSOR_FLOAT32
	default:
		return ti
	}
	if ti == nil {
		return &aiv26.TensorInfo{Dtype: want}
	}
	return &aiv26.TensorInfo{Name: ti.GetName(), Shape: ti.GetShape(), Dtype: want}
}

func bytesToFloat32(b []byte) []float32 {
	if len(b)%4 != 0 {
		return nil
	}
	f := make([]float32, len(b)/4)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return f
}
