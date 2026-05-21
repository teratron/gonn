package quantization

import (
	"math"
	"testing"
)

// TestGoldenValues_A01A02 verifies the L1 §4.2 normative anchor values.
// This is a smoke check for T-19A01/A02; full coverage is in T-19T01.
func TestGoldenValues_A01A02(t *testing.T) {
	cfg := DefaultQuantizationConfig[float64]()
	if cfg.Mode != WeightOnly {
		t.Errorf("DefaultConfig.Mode=%v want WeightOnly", cfg.Mode)
	}
	if cfg.WeightGranularity != PerChannel {
		t.Errorf("DefaultConfig.WeightGranularity=%v want PerChannel", cfg.WeightGranularity)
	}
	if cfg.Strategy != Percentile99p9 {
		t.Errorf("DefaultConfig.Strategy=%v want Percentile99p9", cfg.Strategy)
	}
	if cfg.PercentileThreshold != 99.9 {
		t.Errorf("DefaultConfig.PercentileThreshold=%v want 99.9", cfg.PercentileThreshold)
	}
	if cfg.MinCalibSamples != 32 {
		t.Errorf("DefaultConfig.MinCalibSamples=%v want 32", cfg.MinCalibSamples)
	}

	// L1 §4.2 normative anchor: computeSymmetric(0.0, 3.0)
	p := computeSymmetric(0.0, 3.0)
	wantScale := 3.0 / 127.0
	if math.Abs(p.Scale[0]-wantScale) > 1e-9 {
		t.Errorf("computeSymmetric scale=%v want≈%v", p.Scale[0], wantScale)
	}
	if p.ZeroPoint[0] != 0 {
		t.Errorf("computeSymmetric zp=%v want 0", p.ZeroPoint[0])
	}

	// quantize(1.0) must be 42 per L1 §4.2
	got := quantize(1.0, p, 0)
	if got != 42 {
		t.Errorf("quantize(1.0)=%v want 42", got)
	}

	// dequantize(42) ≈ 0.9921 (≤0.01 abs err)
	r := dequantize(42, p, 0)
	if math.Abs(r-1.0) > 0.01 {
		t.Errorf("dequantize(42)=%v abs err from 1.0 = %v > 0.01", r, math.Abs(r-1.0))
	}

	// roundHalfEven table from L1 §4.2
	cases := [][2]int64{{0, 0}, {1, 2}, {2, 2}, {3, 4}}
	for _, c := range cases {
		in := float64(c[0]) + 0.5
		got := roundHalfEven(in)
		if got != c[1] {
			t.Errorf("roundHalfEven(%v)=%v want %v", in, got, c[1])
		}
	}
}
