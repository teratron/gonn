package optimizer

import (
	"math"
	"testing"
)

func TestClipByGlobalNorm(t *testing.T) {
	t.Parallel()

	t.Run("clips_when_norm_exceeds_threshold", func(t *testing.T) {
		t.Parallel()
		// ||g||₂ = sqrt(3²+4²) = 5; threshold = 2.5 → scale = 0.5
		grads := [][]float64{{3, 4}}
		ClipByGlobalNorm(grads, 2.5)
		want := []float64{1.5, 2.0}
		for i, v := range grads[0] {
			if math.Abs(v-want[i]) > 1e-10 {
				t.Errorf("grads[0][%d] = %v, want %v", i, v, want[i])
			}
		}
	})

	t.Run("no_clip_when_norm_below_threshold", func(t *testing.T) {
		t.Parallel()
		grads := [][]float64{{1, 1}}
		ClipByGlobalNorm(grads, 10.0)
		if math.Abs(grads[0][0]-1) > 1e-12 || math.Abs(grads[0][1]-1) > 1e-12 {
			t.Errorf("grads modified unexpectedly: %v", grads[0])
		}
	})

	t.Run("invariant_norm_le_threshold", func(t *testing.T) {
		t.Parallel()
		// 5 random ||g|| values including ones below threshold.
		cases := []struct {
			threshold float64
			grads     []float64
		}{
			{1.0, []float64{0.9, 0.0}},           // norm < threshold → no-op
			{1.0, []float64{3, 4, 0}},            // norm = 5 > 1
			{2.0, []float64{1, 1, 1, 1}},         // norm = 2 = threshold → no-op
			{0.5, []float64{0.3, 0.4}},           // norm = 0.5 = threshold → no-op
			{1.0, []float64{100, 200, 300, 400}}, // large norm
		}
		for _, tc := range cases {
			g := make([]float64, len(tc.grads))
			copy(g, tc.grads)
			ClipByGlobalNorm([][]float64{g}, tc.threshold)

			var sumSq float64
			for _, v := range g {
				sumSq += v * v
			}
			norm := math.Sqrt(sumSq)
			if norm > tc.threshold+1e-9 {
				t.Errorf("threshold=%.2f: norm after clip = %.6f > threshold", tc.threshold, norm)
			}
		}
	})

	t.Run("noop_on_zero_threshold", func(t *testing.T) {
		t.Parallel()
		grads := [][]float64{{1, 2, 3}}
		ClipByGlobalNorm(grads, 0.0)
		for i, v := range []float64{1, 2, 3} {
			if grads[0][i] != v {
				t.Errorf("zero threshold should not modify grads")
			}
		}
	})

	t.Run("multi_slice", func(t *testing.T) {
		t.Parallel()
		// Global norm across two slices: ||[3,4]||=5, ||[0]||=0 → total norm=5.
		g1 := []float64{3, 4}
		g2 := []float64{0}
		ClipByGlobalNorm([][]float64{g1, g2}, 2.5)
		var sumSq float64
		for _, v := range g1 {
			sumSq += v * v
		}
		for _, v := range g2 {
			sumSq += v * v
		}
		norm := math.Sqrt(sumSq)
		if norm > 2.5+1e-9 {
			t.Errorf("multi-slice clip failed: norm = %v", norm)
		}
	})
}

func TestClipByGlobalNormInvariant(t *testing.T) {
	t.Parallel()
	const threshold = 1.0
	// 5 random gradient configurations — after clip, norm ≤ threshold.
	configs := [][]float64{
		{10, 0, 0},
		{3, 4},
		{0.5, 0.5},
		{100, 200, 300},
		{1e-8, 1e-8},
	}
	for _, cfg := range configs {
		g := make([]float64, len(cfg))
		copy(g, cfg)
		ClipByGlobalNorm([][]float64{g}, threshold)
		var sumSq float64
		for _, v := range g {
			sumSq += v * v
		}
		if math.Sqrt(sumSq) > threshold+1e-9 {
			t.Errorf("config %v: norm after clip = %v > %v", cfg, math.Sqrt(sumSq), threshold)
		}
	}
}
