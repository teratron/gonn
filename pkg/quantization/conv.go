package quantization

import (
	convpkg "github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/utils"
)

// convPadMode aliases conv.PadMode so persistence.go can reference it
// without importing the conv package again.
type convPadMode = convpkg.PadMode

// QuantizedConv1D is a weight-only int8 quantized 1-D convolutional layer.
// Weight layout: NumFilters × KernelSize (filter-major, same as Conv1D Variant A).
// Forward dequantizes filter weights on the fly and computes float64 accumulation.
//
// AI-Meta:
//   - Purpose: Weight-only int8 Conv1D layer for Phase α inference (T-19A05).
//   - Stability: Experimental.
type QuantizedConv1D[T utils.Float] struct {
	Weights      []int8
	Bias         []float64
	WeightParams QuantizationParams
	NumFilters   int
	KernelSize   int
	InLen        int
	Stride       int
	Padding      convPadMode
	UseBias      bool
}

// Forward applies int8 weight-only cross-correlation.
// Output is filter-major: filter f spans output[f*outPer : (f+1)*outPer].
func (c *QuantizedConv1D[T]) Forward(x []T) []T {
	inLen := len(x)
	outPer := convpkg.OutputLen(inLen, c.KernelSize, c.Stride, c.Padding)
	if outPer == 0 {
		return []T{}
	}
	out := make([]T, outPer*c.NumFilters)

	var padL int
	if c.Padding == convpkg.PadSame {
		padL, _ = convpkg.PadSamePadding(inLen, c.KernelSize, c.Stride)
	}

	for f := range c.NumFilters {
		wBase := f * c.KernelSize
		oBase := f * outPer
		for i := range outPer {
			var acc float64
			start := i*c.Stride - padL
			for k := range c.KernelSize {
				pos := start + k
				if pos < 0 || pos >= inLen {
					continue
				}
				w := dequantize(c.Weights[wBase+k], c.WeightParams, f)
				acc += w * float64(x[pos])
			}
			if c.UseBias && len(c.Bias) > f {
				acc += c.Bias[f]
			}
			out[oBase+i] = T(acc)
		}
	}
	return out
}

func newQuantizedConv1D[T utils.Float](src *convpkg.Conv1D[T], cfg QuantizationConfig[T]) *QuantizedConv1D[T] {
	wf64 := toFloat64Slice(src.Weights)
	qw, params := quantizeWeightsTensor(wf64, src.NumFilters, src.KernelSize, cfg.WeightGranularity)
	params.Strategy = cfg.Strategy

	var bias []float64
	if src.UseBias {
		bias = toFloat64Slice(src.Biases)
	}

	return &QuantizedConv1D[T]{
		WeightParams: params,
		Weights:      qw,
		Bias:         bias,
		NumFilters:   src.NumFilters,
		KernelSize:   src.KernelSize,
		InLen:        src.InLen,
		Stride:       src.Stride,
		Padding:      src.Padding,
		UseBias:      src.UseBias,
	}
}

// QuantizedConv2D is a weight-only int8 quantized 2-D convolutional layer.
// Weight layout: NumFilters × InChannels × KernelH × KernelW (filter-major CHW,
// same as Conv2D CONV2D-C9 convention).
// Forward dequantizes filter weights on the fly and computes float64 accumulation.
//
// AI-Meta:
//   - Purpose: Weight-only int8 Conv2D layer for Phase α inference (T-19A05).
//   - Stability: Experimental.
type QuantizedConv2D[T utils.Float] struct {
	Weights      []int8
	Bias         []float64
	WeightParams QuantizationParams
	NumFilters   int
	InChannels   int
	KernelH      int
	KernelW      int
	InLen        int
	StrideH      int
	StrideW      int
	Padding      convPadMode
	UseBias      bool
}

// Forward applies int8 weight-only 2-D cross-correlation over a CHW-flat input.
// Output is filter-major CHW: filter f spans output[f*outH*outW : (f+1)*outH*outW].
func (c *QuantizedConv2D[T]) Forward(x []T) []T {
	// Infer InH, InW from InLen and InChannels.
	totalSpatial := len(x) / c.InChannels
	if len(x) == 0 || c.InChannels <= 0 || totalSpatial*c.InChannels != len(x) {
		return []T{}
	}
	inH := convpkg.Isqrt(totalSpatial)
	inW := totalSpatial / inH
	if inH <= 0 || inH*inW != totalSpatial {
		return []T{}
	}

	outH := convpkg.OutputLen(inH, c.KernelH, c.StrideH, c.Padding)
	outW := convpkg.OutputLen(inW, c.KernelW, c.StrideW, c.Padding)
	if outH == 0 || outW == 0 {
		return []T{}
	}

	out := make([]T, c.NumFilters*outH*outW)

	var padTop, padLeft int
	if c.Padding == convpkg.PadSame {
		padTop, _ = convpkg.PadSamePadding(inH, c.KernelH, c.StrideH)
		padLeft, _ = convpkg.PadSamePadding(inW, c.KernelW, c.StrideW)
	}

	kHW := c.KernelH * c.KernelW
	cKHW := c.InChannels * kHW
	hw := inH * inW
	oHW := outH * outW

	for f := range c.NumFilters {
		wfBase := f * cKHW
		ofBase := f * oHW
		for oy := range outH {
			for ox := range outW {
				var acc float64
				iyBase := oy*c.StrideH - padTop
				ixBase := ox*c.StrideW - padLeft
				for ch := range c.InChannels {
					wcBase := wfBase + ch*kHW
					xcBase := ch * hw
					for ky := range c.KernelH {
						iy := iyBase + ky
						if iy < 0 || iy >= inH {
							continue
						}
						wkyBase := wcBase + ky*c.KernelW
						xyBase := xcBase + iy*inW
						for kx := range c.KernelW {
							ix := ixBase + kx
							if ix < 0 || ix >= inW {
								continue
							}
							w := dequantize(c.Weights[wkyBase+kx], c.WeightParams, f)
							acc += w * float64(x[xyBase+ix])
						}
					}
				}
				if c.UseBias && len(c.Bias) > f {
					acc += c.Bias[f]
				}
				out[ofBase+oy*outW+ox] = T(acc)
			}
		}
	}
	return out
}

func newQuantizedConv2D[T utils.Float](src *convpkg.Conv2D[T], cfg QuantizationConfig[T]) *QuantizedConv2D[T] {
	cin := src.InChannels * src.KernelH * src.KernelW
	wf64 := toFloat64Slice(src.Weights)
	qw, params := quantizeWeightsTensor(wf64, src.NumFilters, cin, cfg.WeightGranularity)
	params.Strategy = cfg.Strategy

	var bias []float64
	if src.UseBias {
		bias = toFloat64Slice(src.Biases)
	}

	inLen := src.InH * src.InW
	if inLen == 0 {
		inLen = src.InputSize() / src.InChannels
	}

	return &QuantizedConv2D[T]{
		WeightParams: params,
		Weights:      qw,
		Bias:         bias,
		NumFilters:   src.NumFilters,
		InChannels:   src.InChannels,
		KernelH:      src.KernelH,
		KernelW:      src.KernelW,
		InLen:        inLen,
		StrideH:      src.StrideH,
		StrideW:      src.StrideW,
		Padding:      src.Padding,
		UseBias:      src.UseBias,
	}
}
