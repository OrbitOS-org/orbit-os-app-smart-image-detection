package main

import (
	"fmt"
	"math"
	"sort"
)

var coco80Names = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
	"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
	"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

func className(id int) string {
	if id >= 0 && id < len(coco80Names) {
		return coco80Names[id]
	}
	return fmt.Sprintf("class_%d", id)
}

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

type detection struct {
	classID int
	score   float32
	x1, y1  float32
	x2, y2  float32
}

// DetectionJSON is one box for API responses.
type DetectionJSON struct {
	Class string    `json:"class"`
	Score float32   `json:"score"`
	Box   []float32 `json:"box"` // normalized xyxy [x1,y1,x2,y2] in 0–1
}

const minLogitFilterOff = float32(-1e8)

// classScoresLookLikeProbabilities is a heuristic: some TFLite exports already apply sigmoid to the class scores.
func classScoresLookLikeProbabilities(data []float32, n, nc int) bool {
	var maxV float32 = -1e9
	var minV float32 = 1e9
	for j := 0; j < n; j++ {
		for k := 0; k < nc; k++ {
			v := data[(4+k)*n+j]
			if v > maxV {
				maxV = v
			}
			if v < minV {
				minV = v
			}
		}
	}
	return maxV <= 1.0 && minV >= -0.01 && maxV > 1e-6
}

func decodeYOLOv8(data []float32, shape []int32, confThresh, iouThresh, minLogit float32) []detection {
	if len(shape) != 3 {
		return nil
	}
	b, c, n := int(shape[0]), int(shape[1]), int(shape[2])
	if b != 1 || c < 5 || n < 1 {
		return nil
	}
	nc := c - 4
	if nc < 1 {
		return nil
	}

	probMode := classScoresLookLikeProbabilities(data, n, nc)

	var raw []detection
	for j := 0; j < n; j++ {
		cx := data[0*n+j]
		cy := data[1*n+j]
		w := data[2*n+j]
		h := data[3*n+j]
		bestCl := 0
		var maxLogit float32
		for k := 0; k < nc; k++ {
			v := data[(4+k)*n+j]
			if k == 0 || v > maxLogit {
				maxLogit = v
				bestCl = k
			}
		}
		if minLogit > minLogitFilterOff && maxLogit < minLogit {
			continue
		}
		var bestScore float32
		if probMode {
			bestScore = maxLogit
		} else {
			bestScore = sigmoid(maxLogit)
		}
		if bestScore < confThresh {
			continue
		}
		x1 := cx - w*0.5
		y1 := cy - h*0.5
		x2 := cx + w*0.5
		y2 := cy + h*0.5
		aw := x2 - x1
		ah := y2 - y1
		if aw < 0.02 || ah < 0.02 || x1 >= x2 || y1 >= y2 {
			continue
		}
		raw = append(raw, detection{
			classID: bestCl,
			score:   bestScore,
			x1:      x1, y1: y1, x2: x2, y2: y2,
		})
	}
	return nms(raw, iouThresh)
}

func iou(a, b detection) float32 {
	ix1 := max32(a.x1, b.x1)
	iy1 := max32(a.y1, b.y1)
	ix2 := min32(a.x2, b.x2)
	iy2 := min32(a.y2, b.y2)
	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	areaA := (a.x2 - a.x1) * (a.y2 - a.y1)
	areaB := (b.x2 - b.x1) * (b.y2 - b.y1)
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func nms(d []detection, iouTh float32) []detection {
	sort.Slice(d, func(i, j int) bool { return d[i].score > d[j].score })
	var keep []detection
	for _, a := range d {
		sup := false
		for _, b := range keep {
			if a.classID == b.classID && iou(a, b) > iouTh {
				sup = true
				break
			}
		}
		if !sup {
			keep = append(keep, a)
		}
	}
	return keep
}

func transpose8400x84(data []float32, c int) []float32 {
	const n = 8400
	out := make([]float32, n*c)
	for i := 0; i < n; i++ {
		for j := 0; j < c; j++ {
			out[j*n+i] = data[i*c+j]
		}
	}
	return out
}

// detectionsToJSON runs the same decode path as ai_test's printYOLODetections.
func detectionsToJSON(shape []int32, data []float32, confThresh, iouThresh, minLogit float32) []DetectionJSON {
	if len(shape) != 3 {
		return nil
	}
	d := data
	s := shape
	if shape[1] == 8400 && shape[2] >= 5 {
		d = transpose8400x84(data, int(shape[2]))
		s = []int32{1, shape[2], 8400}
	}
	raw := decodeYOLOv8(d, s, confThresh, iouThresh, minLogit)
	out := make([]DetectionJSON, 0, len(raw))
	for _, x := range raw {
		out = append(out, DetectionJSON{
			Class: className(x.classID),
			Score: x.score,
			Box:   []float32{x.x1, x.y1, x.x2, x.y2},
		})
	}
	return out
}
