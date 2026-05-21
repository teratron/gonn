package quantization

import "math"

// Granularity selects the scale granularity for weight or activation quantization.
//
// AI-Meta:
//   - Purpose: Enum selecting per-tensor vs per-channel quantization granularity.
//   - Implementations: PerTensor, PerChannel.
type Granularity int

const (
	// PerTensor uses a single scale/zero_point for the whole tensor.
	PerTensor Granularity = iota
	// PerChannel uses one scale/zero_point per output channel (QUANT-3, QUANT-C4).
	PerChannel
)

// CalibStrategy selects the activation-range estimation algorithm used by
// CalibrationRunner.
//
// AI-Meta:
//   - Purpose: Enum selecting calibration range-estimation strategy.
//   - Implementations: MinMax, Percentile99p9, Entropy.
type CalibStrategy int

const (
	// MinMax uses the observed min/max of activation values.
	MinMax CalibStrategy = iota
	// Percentile99p9 clips outliers at the 99.9th percentile (default).
	Percentile99p9
	// Entropy minimises KL-divergence between float and quantized distributions (Phase β+).
	Entropy
)

// QuantizationParams stores the per-layer (or per-channel) affine parameters
// that map int8 quantized values back to float activations or weights.
//
// For PerTensor: len(Scale)==1 and len(ZeroPoint)==1.
// For PerChannel: len(Scale)==Cout and len(ZeroPoint)==Cout.
//
// AI-Meta:
//   - Purpose: Affine scale+zero_point metadata for a single quantized tensor or weight matrix.
type QuantizationParams struct {
	Scale       []float64
	ZeroPoint   []int32
	Granularity Granularity
	Strategy    CalibStrategy
}

// dequantize reconstructs a float64 value from an int8 quantized value:
//
//	r = scale[ch] · (q − zero_point[ch])
func dequantize(q int8, p QuantizationParams, ch int) float64 {
	idx := channelIndex(p, ch)
	return p.Scale[idx] * (float64(q) - float64(p.ZeroPoint[idx]))
}

// quantize maps a float64 value to int8 using banker's rounding (roundHalfEven)
// and clamping to the valid int8 range [-128, 127].
func quantize(r float64, p QuantizationParams, ch int) int8 {
	idx := channelIndex(p, ch)
	q := roundHalfEven(r/p.Scale[idx] + float64(p.ZeroPoint[idx]))
	switch {
	case q < -128:
		return -128
	case q > 127:
		return 127
	default:
		return int8(q)
	}
}

// roundHalfEven rounds x to the nearest integer using banker's rounding
// (round-half-to-even). When x has a fractional part of exactly 0.5 the
// result is rounded to the nearest even integer, breaking ties toward zero.
//
// L1 §4.2 table: 0.5→0, 1.5→2, 2.5→2, 3.5→4.
func roundHalfEven(x float64) int64 {
	f := math.Floor(x)
	if x-f == 0.5 {
		fi := int64(f)
		if fi%2 == 0 {
			return fi // floor is even: round down
		}
		return fi + 1 // floor is odd: round up
	}
	return int64(math.Round(x))
}

// computeSymmetric returns PerTensor symmetric QuantizationParams for
// the range [rMin, rMax], reserving the -128 slot (range [-127, 127]).
//
//	scale = max(|rMin|, rMax) / 127;  zero_point = 0.
//
// Degenerate guard: rMax−rMin < 1e-6 → scale=1.0, zero_point=0.
func computeSymmetric(rMin, rMax float64) QuantizationParams {
	if rMax-rMin < 1e-6 {
		return degenerateParams()
	}
	absMax := rMax
	if math.Abs(rMin) > absMax {
		absMax = math.Abs(rMin)
	}
	scale := absMax / 127.0
	if scale == 0 {
		scale = 1.0
	}
	return QuantizationParams{
		Scale:       []float64{scale},
		ZeroPoint:   []int32{0},
		Granularity: PerTensor,
	}
}

// computeAsymmetric returns PerTensor asymmetric QuantizationParams for
// the range [rMin, rMax], using the full [-128, 127] int8 range.
//
//	scale = (rMax−rMin) / 255;  zero_point = round(−rMin/scale) − 128.
//
// Degenerate guard: rMax−rMin < 1e-6 → scale=1.0, zero_point=0.
func computeAsymmetric(rMin, rMax float64) QuantizationParams {
	if rMax-rMin < 1e-6 {
		return degenerateParams()
	}
	scale := (rMax - rMin) / 255.0
	zp := int32(math.Round(-rMin/scale)) - 128
	return QuantizationParams{
		Scale:       []float64{scale},
		ZeroPoint:   []int32{zp},
		Granularity: PerTensor,
	}
}

// channelIndex returns the params index for a given channel, collapsing to 0
// for PerTensor granularity.
func channelIndex(p QuantizationParams, ch int) int {
	if p.Granularity == PerTensor {
		return 0
	}
	return ch
}

// degenerateParams returns safe fallback params (scale=1.0, zp=0) for
// degenerate ranges where rMax−rMin < 1e-6.
func degenerateParams() QuantizationParams {
	return QuantizationParams{
		Scale:       []float64{1.0},
		ZeroPoint:   []int32{0},
		Granularity: PerTensor,
	}
}
