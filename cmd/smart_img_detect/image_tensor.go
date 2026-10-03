package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	idraw "image/draw"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"

	aiv26 "github.com/OrbitOS-org/orbit-os-sdk-go/v26/api/ai_service/v26"
	xdraw "golang.org/x/image/draw"
)

// ultralyticsPadRGB is the default letterbox padding colour (same as Ultralytics YOLOv8).
var ultralyticsPadRGB = color.RGBA{R: 114, G: 114, B: 114, A: 255}

// InputTensor holds the preprocessed image ready to pass to AIManager.Infer.
type InputTensor struct {
	Name  string
	Data  []byte
	Shape []int32
	Dtype aiv26.TensorDataType
}

// buildImageInputTensor decodes the image at path (same as ai_test).
func buildImageInputTensor(path string, ti *aiv26.TensorInfo, letterbox bool) (*InputTensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return buildImageInputTensorFromReader(f, ti, letterbox)
}

// buildImageInputTensorFromReader decodes an image from r (web upload, stdin, etc.).
func buildImageInputTensorFromReader(r io.Reader, ti *aiv26.TensorInfo, letterbox bool) (*InputTensor, error) {
	src, _, err := image.Decode(r)
	if err != nil {
		return nil, err
	}
	return buildInputTensorFromImage(src, ti, letterbox)
}

func buildInputTensorFromImage(src image.Image, ti *aiv26.TensorInfo, letterbox bool) (*InputTensor, error) {
	name := "input"
	if ti != nil && ti.GetName() != "" {
		name = ti.GetName()
	}

	shape := []int32{1, 640, 640, 3}
	if ti != nil && len(ti.GetShape()) >= 4 {
		shape = normalizeShape(ti.GetShape())
	}

	dtype := aiv26.TensorDataType_TENSOR_FLOAT32
	if ti != nil {
		d := ti.GetDtype()
		if d != aiv26.TensorDataType_TENSOR_FLOAT32 && d != aiv26.TensorDataType_TENSOR_UINT8 {
			return nil, fmt.Errorf("unsupported model input dtype %v (need FLOAT32 or UINT8)", d)
		}
		dtype = d
	}

	h, w, nhwc := inferLayout(shape)
	var dst *image.RGBA
	if letterbox {
		dst = letterboxResize(src, w, h, ultralyticsPadRGB)
	} else {
		dst = image.NewRGBA(image.Rect(0, 0, w, h))
		xdraw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), xdraw.Over, nil)
	}

	switch dtype {
	case aiv26.TensorDataType_TENSOR_UINT8:
		data, sh, err := flattenUint8(dst, shape, nhwc)
		if err != nil {
			return nil, err
		}
		return &InputTensor{Name: name, Data: data, Shape: sh, Dtype: dtype}, nil
	default:
		data, sh, err := flattenFloat32Bytes(dst, shape, nhwc)
		if err != nil {
			return nil, err
		}
		return &InputTensor{Name: name, Data: data, Shape: sh, Dtype: dtype}, nil
	}
}

func normalizeShape(shape []int32) []int32 {
	s := append([]int32(nil), shape...)
	for i := range s {
		if s[i] > 0 {
			continue
		}
		if i == 0 {
			s[i] = 1
		} else {
			s[i] = 640
		}
	}
	return s
}

func inferLayout(s []int32) (h, w int, nhwc bool) {
	if len(s) != 4 {
		return 640, 640, true
	}
	if s[1] == 3 && s[3] != 3 {
		return int(s[2]), int(s[3]), false
	}
	if s[3] == 3 {
		return int(s[1]), int(s[2]), true
	}
	return int(s[1]), int(s[2]), true
}

func flattenFloat32Bytes(rgba *image.RGBA, shape []int32, nhwc bool) ([]byte, []int32, error) {
	var f32 []float32
	var err error
	if nhwc {
		f32, err = flattenNHWCFloat(rgba, shape)
	} else {
		f32, err = flattenNCHWFloat(rgba, shape)
	}
	if err != nil {
		return nil, nil, err
	}
	return float32sToBytes(f32), shape, nil
}

func flattenUint8(rgba *image.RGBA, shape []int32, nhwc bool) ([]byte, []int32, error) {
	var out []byte
	var err error
	if nhwc {
		out, err = flattenNHWCUint8(rgba, shape)
	} else {
		out, err = flattenNCHWUint8(rgba, shape)
	}
	return out, shape, err
}

func flattenNHWCFloat(rgba *image.RGBA, shape []int32) ([]float32, error) {
	b, h, w := int(shape[0]), int(shape[1]), int(shape[2])
	if shape[3] != 3 {
		return nil, fmt.Errorf("expected channel dim 3, got shape %v", shape)
	}
	out := make([]float32, b*h*w*3)
	idx := 0
	for by := 0; by < b; by++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r16, g16, b16, _ := rgba.At(x, y).RGBA()
				out[idx+0] = float32(r16>>8) / 255.0
				out[idx+1] = float32(g16>>8) / 255.0
				out[idx+2] = float32(b16>>8) / 255.0
				idx += 3
			}
		}
	}
	return out, nil
}

func flattenNCHWFloat(rgba *image.RGBA, shape []int32) ([]float32, error) {
	b, c, h, w := int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
	if c != 3 {
		return nil, fmt.Errorf("expected 3 input channels, got shape %v", shape)
	}
	out := make([]float32, b*c*h*w)
	i := 0
	for bi := 0; bi < b; bi++ {
		for ch := 0; ch < 3; ch++ {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					r16, g16, b16, _ := rgba.At(x, y).RGBA()
					v := []uint32{r16 >> 8, g16 >> 8, b16 >> 8}
					out[i] = float32(v[ch]) / 255.0
					i++
				}
			}
		}
	}
	return out, nil
}

func flattenNHWCUint8(rgba *image.RGBA, shape []int32) ([]byte, error) {
	b, h, w := int(shape[0]), int(shape[1]), int(shape[2])
	if shape[3] != 3 {
		return nil, fmt.Errorf("expected channel dim 3, got shape %v", shape)
	}
	out := make([]byte, b*h*w*3)
	idx := 0
	for by := 0; by < b; by++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r16, g16, b16, _ := rgba.At(x, y).RGBA()
				out[idx+0] = byte(r16 >> 8)
				out[idx+1] = byte(g16 >> 8)
				out[idx+2] = byte(b16 >> 8)
				idx += 3
			}
		}
	}
	return out, nil
}

func flattenNCHWUint8(rgba *image.RGBA, shape []int32) ([]byte, error) {
	b, c, h, w := int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
	if c != 3 {
		return nil, fmt.Errorf("expected 3 input channels, got shape %v", shape)
	}
	out := make([]byte, b*c*h*w)
	i := 0
	for bi := 0; bi < b; bi++ {
		for ch := 0; ch < 3; ch++ {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					r16, g16, b16, _ := rgba.At(x, y).RGBA()
					v := []byte{byte(r16 >> 8), byte(g16 >> 8), byte(b16 >> 8)}
					out[i] = v[ch]
					i++
				}
			}
		}
	}
	return out, nil
}

func letterboxResize(src image.Image, outW, outH int, pad color.RGBA) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, outW, outH))
	idraw.Draw(dst, dst.Bounds(), image.NewUniform(pad), image.Point{}, idraw.Src)
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw <= 0 || sh <= 0 {
		return dst
	}
	scale := math.Min(float64(outW)/float64(sw), float64(outH)/float64(sh))
	nw := int(math.Round(float64(sw) * scale))
	nh := int(math.Round(float64(sh) * scale))
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	if nw > outW {
		nw = outW
	}
	if nh > outH {
		nh = outH
	}
	scaled := image.NewRGBA(image.Rect(0, 0, nw, nh))
	xdraw.CatmullRom.Scale(scaled, scaled.Bounds(), src, b, xdraw.Over, nil)
	ox := (outW - nw) / 2
	oy := (outH - nh) / 2
	r := image.Rect(ox, oy, ox+nw, oy+nh)
	idraw.Draw(dst, r, scaled, scaled.Bounds().Min, idraw.Over)
	return dst
}

func float32sToBytes(f []float32) []byte {
	b := make([]byte, len(f)*4)
	for i, v := range f {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}
