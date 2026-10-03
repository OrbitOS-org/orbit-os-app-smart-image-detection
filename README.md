<p align="center">
  <img src="https://www.orbit-os.org/images/vscode/orbit-os-logo.png" width="300" alt="Orbit OS">
</p>

<h1 align="center">Edge AI – Smart Image Detection for Orbit OS</h1>

<p align="center"><b>Object detection on your own images, running on the device — an example of the Orbit OS AI API.</b></p>

An [Orbit OS](https://www.orbit-os.org/?ref=github-smart-img-detect) app that detects and labels objects in your own images. Upload a JPEG or PNG, press **Detect Objects**, and the app draws the bounding boxes and lists what it found, with timing for each run. Inference runs **locally on the device** with a YOLOv8 TensorFlow Lite model through the Orbit OS AI service — no cloud API.

It is also a practical, readable example of **Edge AI on Orbit OS**: how an app loads a model, runs inference and decodes the result with the [Orbit OS Go SDK](https://github.com/OrbitOS-org/orbit-os-sdk-go).

Runs on Raspberry Pi, Arduino UNO Q and other ARM64 devices with Orbit OS (free Community Edition).

## Features

- **Detect objects in your images** — upload a JPEG or PNG, get bounding boxes, labels and scores (80 COCO classes)
- **On-device inference** through the Orbit OS AI service (TFLite) — the image never leaves the device
- **Model included** — a YOLOv8n model ships inside the app, so it works right after install
- **Timing for each run** — round-trip time and on-device inference time
- **Raw JSON** response available for inspection
- Web UI served through the Orbit OS AppHub, behind the device login

## Install

**From the Orbit OS Store (recommended):** install [Edge AI – Smart Image Detection](https://store.orbit-os.org/app/smart-img-detect?ref=github-smart-img-detect) on your device in one click.

<a href="https://store.orbit-os.org/app/smart-img-detect?ref=github-smart-img-detect"><img src="https://www.orbit-os.org/images/badges/get-it-on-orbit-os-store@3x.png" width="200" alt="Get it on Orbit OS Store"></a>

**From source — recommended: [Orbit Studio](https://marketplace.visualstudio.com/items?itemName=orbit-os.orbit-studio) (VS Code):**

You need [VS Code](https://code.visualstudio.com/) with the Orbit Studio extension and **[Go](https://go.dev/dl/) 1.25 or newer** installed (`go` on your PATH).

1. Clone the repository and open the folder in VS Code with the Orbit Studio extension:
   ```bash
   git clone https://github.com/OrbitOS-org/orbit-os-app-smart-image-detection
   code orbit-os-app-smart-image-detection
   ```
2. In the Orbit sidebar, run **Add / Update SDK** and set your device's IP.
3. Use **Run** to try it live against a device in Developer Mode, then **Build + Deploy** to install the signed `.orb`.

**Without Orbit Studio:** `go build ./cmd/smart_img_detect` builds the binary with the published SDK module — use Orbit Studio to package and sign the `.orb`.

## Getting started

1. Open **Edge AI – Smart Image Detection** from the AppHub on your device.
2. Drop an image (JPEG or PNG) or browse for one.
3. Press **Detect Objects** — the boxes, labels, scores and timing appear; *Show JSON* reveals the raw response.

## Model

The app ships with one model, [`cmd/smart_img_detect/orb/data/models/yolov8_spotlab.tflite`](cmd/smart_img_detect/orb/data/models/yolov8_spotlab.tflite): **YOLOv8n** by [Ultralytics](https://github.com/ultralytics/ultralytics), trained on COCO (80 classes, 640×640 input), exported to TensorFlow Lite. Orbit Studio packages everything under `orb/data/` into the `.orb`, and the app loads it from `models/` in its data folder at startup.

The model is licensed by Ultralytics under **AGPL-3.0**, which is why this repository is AGPL-3.0 too — see [License](#license).

## How it works

The whole AI flow is three SDK calls — the rest of the code is image preparation and decoding the output tensor:

```go
client, _ := orbitos.NewClientAuto(host) // Unix socket on the device, TCP + mTLS from a laptop
defer client.Close()

// 1. Upload the model to the device's AI service and load it
model, _ := client.AIManager.UploadAndLoadModel("yolo8-spotlab", "models/yolov8_spotlab.tflite",
	aiv26.ModelBackend_TFLITE, aiv26.ExecutionMode_EXEC_CPU)

// 2. Run inference with the image as an input tensor
resp, _ := model.Infer(ctx, tensor.Data, tensor.Shape, tensor.Dtype)

// 3. Free it when you're done
model.Unload()
```

| File | What |
|---|---|
| `main.go` | HTTP server, AppHub registration, model loading and calls to the AI service |
| `image_tensor.go` | decodes the image, letterbox-resizes it and builds the input tensor (NHWC/NCHW, uint8/float32) |
| `yolo_decode.go` | turns the output tensor into boxes, classes and scores (NMS included) |
| `static/` | web UI |

## Development (Orbit Studio)

This project follows the standard [Orbit Studio](https://marketplace.visualstudio.com/items?itemName=orbit-os.orbit-studio) layout:

| Path | What |
|---|---|
| `cmd/smart_img_detect/` | app source, `metadata.json` (manifest & permissions) |
| `cmd/smart_img_detect/orb/icon.svg` | launcher / Store icon |
| `cmd/smart_img_detect/orb/data/models/` | the model packaged into the `.orb` |
| `orbit.project.json` | Orbit Studio project settings |

- **Recommended workflow:** open the folder in VS Code with Orbit Studio, **Add / Update SDK** (creates the local `orbit-os-sdk-go/` copy and `go.work`, both git-ignored), then **Run** to develop against a device in Developer Mode, or **Build + Deploy** to install the `.orb`.
- Without Orbit Studio, `go build` uses the published SDK module [`github.com/OrbitOS-org/orbit-os-sdk-go/v26`](https://pkg.go.dev/github.com/OrbitOS-org/orbit-os-sdk-go/v26). The `-host <DEVICE_IP>` flag is only used when running off-device (development); on the device the SDK uses the local Unix socket. When running off-device, copy the model to `cmd/smart_img_detect/models/` (git-ignored) and start the app from `cmd/smart_img_detect/`.
- Development TLS certificates live in `cmd/certs/grpc/` and are never committed.
- Permissions used: `AiService`, `AppHubService`, `SystemService`.

## Links

[App in the Store](https://store.orbit-os.org/app/smart-img-detect?ref=github-smart-img-detect) · [Orbit OS](https://www.orbit-os.org/?ref=github-smart-img-detect) · [Getting started](https://www.orbit-os.org/getting_started.html?ref=github-smart-img-detect) · [SDK reference](https://www.orbit-os.org/api-reference.html?ref=github-smart-img-detect) · [Forum](https://forum.orbit-os.org/?ref=github-smart-img-detect) · info@orbit-os.org

## Acknowledgments

Object detection uses the **YOLOv8n** model from **[Ultralytics](https://github.com/ultralytics/ultralytics)**. Thanks to the Ultralytics team and contributors for publishing it.

## License

**AGPL-3.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

This app bundles the Ultralytics YOLOv8n model, which is licensed under AGPL-3.0, so the app as a whole is distributed under the same license. The Orbit OS Go SDK it uses is Apache-2.0.
