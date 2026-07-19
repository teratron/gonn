package quantization

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

// QuantizedDense is a weight-only (Phase α) or full-int8 (Phase β) quantized
// fully-connected layer. Weights are stored row-major (Cout×Cin).
//
// Weight-only mode (ActParams.Scale == nil): dequantize weights on the fly,
// float64 dot-product with x.
//
// Full-int8 mode (ActParams.Scale != nil): quantize x to int8, int32 accumulator
// GEMM, per-channel scale dequantize at output.
//
// AI-Meta:
//   - Purpose: Int8 Dense layer for Phase α (weight-only) and Phase β (full-int8) inference (T-19A04/A08).
//   - Stability: Experimental.
type QuantizedDense[T utils.Float] struct {
	Weights      []int8
	Bias         []float64
	WeightParams QuantizationParams
	ActParams    QuantizationParams
	Cout         int
	Cin          int
	// Act is the activation applied to the accumulated output when ApplyAct
	// is true. Set by Quantize for MLP-head layers so the quantized chain
	// reproduces the full float forward (activation.SOFTMAX is applied
	// whole-vector). Zero-value (ApplyAct=false) keeps the raw linear output —
	// the historical behaviour for standalone use.
	Act      activation.Type
	ApplyAct bool
}

// SetActParams wires activation quantization params (called by Quantize[T]
// when cfg.Mode == FullInt8 after CalibrationRunner.Params()).
func (d *QuantizedDense[T]) SetActParams(p QuantizationParams) {
	d.ActParams = p
}

// Forward dispatches to weight-only or full-int8 path depending on ActParams,
// then applies the configured activation when ApplyAct is set.
func (d *QuantizedDense[T]) Forward(x []T) []T {
	var out []T
	if len(d.ActParams.Scale) > 0 {
		out = d.forwardFullInt8(x)
	} else {
		out = d.forwardWeightOnly(x)
	}
	if d.ApplyAct {
		if d.Act == activation.SOFTMAX {
			activation.SoftmaxInto(out, out)
		} else {
			for i, v := range out {
				out[i] = activation.Activation(v, d.Act)
			}
		}
	}
	return out
}

// forwardWeightOnly dequantizes each weight row on the fly, float64 accumulation.
func (d *QuantizedDense[T]) forwardWeightOnly(x []T) []T {
	out := make([]T, d.Cout)
	for c := range d.Cout {
		var acc float64
		for k := range d.Cin {
			w := dequantize(d.Weights[c*d.Cin+k], d.WeightParams, c)
			acc += w * float64(x[k])
		}
		if len(d.Bias) > c {
			acc += d.Bias[c]
		}
		out[c] = T(acc)
	}
	return out
}

// forwardFullInt8 quantizes x to int8 then accumulates in int32.
// Per-channel scale: y[c] = scale_W[c] * scale_x * float64(acc[c]) + bias[c].
// Zero-point correction: acc[c] -= zp_W[c] * sum(x_int8).
func (d *QuantizedDense[T]) forwardFullInt8(x []T) []T {
	xInt8 := make([]int8, d.Cin)
	var xSum int32
	for k := range d.Cin {
		xInt8[k] = quantize(float64(x[k]), d.ActParams, 0)
		xSum += int32(xInt8[k])
	}

	scaleX := d.ActParams.Scale[0]
	zpX := int32(d.ActParams.ZeroPoint[0])

	out := make([]T, d.Cout)
	for c := range d.Cout {
		var acc int32
		for k := range d.Cin {
			acc += int32(d.Weights[c*d.Cin+k]) * int32(xInt8[k])
		}
		// Zero-point bias correction: subtract zp_W[c] * sum(x_int8)
		// and zp_x * sum(W[c,:]).
		zpW := int32(d.WeightParams.ZeroPoint[channelIndex(d.WeightParams, c)])
		acc -= zpW * xSum
		// Subtract zp_x * sum(W[c,:]) term.
		var wSum int32
		for k := range d.Cin {
			wSum += int32(d.Weights[c*d.Cin+k])
		}
		acc -= zpX * wSum
		// Add zp_W * zp_x * Cin correction term.
		acc += zpW * zpX * int32(d.Cin)

		scaleW := d.WeightParams.Scale[channelIndex(d.WeightParams, c)]
		y := scaleW * scaleX * float64(acc)
		if len(d.Bias) > c {
			y += d.Bias[c]
		}
		out[c] = T(y)
	}
	return out
}

func newQuantizedDense[T utils.Float](w []float64, bias []float64, cout, cin int, cfg QuantizationConfig[T]) *QuantizedDense[T] {
	qw, params := quantizeWeightsTensor(w, cout, cin, cfg.WeightGranularity)
	params.Strategy = cfg.Strategy
	return &QuantizedDense[T]{
		WeightParams: params,
		Weights:      qw,
		Bias:         bias,
		Cout:         cout,
		Cin:          cin,
	}
}
