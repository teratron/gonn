package utils

import (
	"math"
	"testing"
)

func TestNewRNGZeroSeedFallsBackToTime(t *testing.T) {
	t.Parallel()
	rng, seed := NewRNG(0)
	if rng == nil {
		t.Fatal("NewRNG returned nil *rand.Rand")
	}
	if seed == 0 {
		t.Errorf("NewRNG(0) must replace zero seed with a non-zero wall-clock value, got %d", seed)
	}
}

func TestNewRNGReproducible(t *testing.T) {
	t.Parallel()
	a, _ := NewRNG(42)
	b, _ := NewRNG(42)
	for i := range 256 {
		x, y := a.Float64(), b.Float64()
		if x != y {
			t.Fatalf("seeded sequences diverge at step %d: %v vs %v", i, x, y)
		}
	}
}

func TestNewRNGDifferentSeedsDiffer(t *testing.T) {
	t.Parallel()
	a, _ := NewRNG(1)
	b, _ := NewRNG(2)
	identical := 0
	for range 256 {
		if a.Float64() == b.Float64() {
			identical++
		}
	}
	// Allow a handful of accidental collisions; total identity is the failure mode.
	if identical > 8 {
		t.Errorf("seeds 1 and 2 produced %d/256 identical samples — entropy too low", identical)
	}
}

func TestXavierUniformDistribution(t *testing.T) {
	t.Parallel()
	rng, _ := NewRNG(1234)
	const fanIn, fanOut, n = 64, 64, 20000
	a := math.Sqrt(6.0 / float64(fanIn+fanOut))

	var sum, sqSum float64
	for range n {
		v := float64(XavierUniform[float64](rng, fanIn, fanOut))
		if v < -a-1e-9 || v > a+1e-9 {
			t.Fatalf("sample %v out of range [-%v, %v]", v, a, a)
		}
		sum += v
		sqSum += v * v
	}
	mean := sum / n
	variance := sqSum/n - mean*mean
	wantVar := (2 * a) * (2 * a) / 12.0 // U[-a, a] variance
	// 5% tolerance — generous for n=20000, deterministic seed.
	if math.Abs(mean) > 0.05*a {
		t.Errorf("XavierUniform mean = %v; want |mean| <= %v", mean, 0.05*a)
	}
	if math.Abs(variance-wantVar)/wantVar > 0.05 {
		t.Errorf("XavierUniform variance = %v; want %v (5%% tolerance)", variance, wantVar)
	}
}

func TestHeNormalDistribution(t *testing.T) {
	t.Parallel()
	rng, _ := NewRNG(1234)
	const fanIn, n = 64, 20000
	wantSigma := math.Sqrt(2.0 / float64(fanIn))

	var sum, sqSum float64
	for range n {
		v := float64(HeNormal[float64](rng, fanIn))
		sum += v
		sqSum += v * v
	}
	mean := sum / n
	variance := sqSum/n - mean*mean
	wantVar := wantSigma * wantSigma
	if math.Abs(mean) > 0.05*wantSigma {
		t.Errorf("HeNormal mean = %v; want |mean| <= %v", mean, 0.05*wantSigma)
	}
	if math.Abs(variance-wantVar)/wantVar > 0.05 {
		t.Errorf("HeNormal variance = %v; want %v (5%% tolerance)", variance, wantVar)
	}
}

func TestUniformDistribution(t *testing.T) {
	t.Parallel()
	rng, _ := NewRNG(1234)
	const n = 20000
	var sum, sqSum float64
	for range n {
		v := float64(Uniform[float64](rng))
		if v < -1 || v >= 1 {
			t.Fatalf("Uniform sample %v outside [-1, 1)", v)
		}
		sum += v
		sqSum += v * v
	}
	mean := sum / n
	variance := sqSum/n - mean*mean
	wantVar := 4.0 / 12.0 // U[-1, 1) variance
	if math.Abs(mean) > 0.05 {
		t.Errorf("Uniform mean = %v; want |mean| <= 0.05", mean)
	}
	if math.Abs(variance-wantVar)/wantVar > 0.05 {
		t.Errorf("Uniform variance = %v; want %v", variance, wantVar)
	}
}

func TestSamplersWithFloat32(t *testing.T) {
	t.Parallel()
	rng, _ := NewRNG(7)
	if v := XavierUniform[float32](rng, 4, 4); math.IsNaN(float64(v)) {
		t.Fatalf("XavierUniform[float32] produced NaN")
	}
	if v := HeNormal[float32](rng, 4); math.IsNaN(float64(v)) {
		t.Fatalf("HeNormal[float32] produced NaN")
	}
	if v := Uniform[float32](rng); math.IsNaN(float64(v)) {
		t.Fatalf("Uniform[float32] produced NaN")
	}
}

func TestSamplersDegenerateFanFallback(t *testing.T) {
	t.Parallel()
	rng, _ := NewRNG(11)
	// Degenerate fan counts must not divide by zero or produce NaN — the
	// fallback to Uniform is the agreed contract (init.go) since callers
	// upstream still get a finite weight to populate the matrix.
	if v := XavierUniform[float64](rng, 0, 0); math.IsNaN(v) || math.IsInf(v, 0) {
		t.Errorf("XavierUniform with zero fan must fall back to Uniform, got %v", v)
	}
	if v := HeNormal[float64](rng, 0); math.IsNaN(v) || math.IsInf(v, 0) {
		t.Errorf("HeNormal with zero fan must fall back to Uniform, got %v", v)
	}
}

func TestSamplersPanicOnNilRNG(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		fn   func()
	}{
		{"XavierUniform", func() { _ = XavierUniform[float64](nil, 4, 4) }},
		{"HeNormal", func() { _ = HeNormal[float64](nil, 4) }},
		{"Uniform", func() { _ = Uniform[float64](nil) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("%s with nil rng must panic", c.name)
				}
			}()
			c.fn()
		})
	}
}

// BenchmarkXavierUniformFloat32 covers the hot path in layer initialization;
// per RULES.md §C30.4 PERF-1 hot paths require a Benchmark*.
func BenchmarkXavierUniformFloat32(b *testing.B) {
	rng, _ := NewRNG(1)
	for b.Loop() {
		_ = XavierUniform[float32](rng, 64, 64)
	}
}

func BenchmarkXavierUniformFloat64(b *testing.B) {
	rng, _ := NewRNG(1)
	for b.Loop() {
		_ = XavierUniform[float64](rng, 64, 64)
	}
}

func BenchmarkHeNormalFloat32(b *testing.B) {
	rng, _ := NewRNG(1)
	for b.Loop() {
		_ = HeNormal[float32](rng, 64)
	}
}

func BenchmarkUniformFloat32(b *testing.B) {
	rng, _ := NewRNG(1)
	for b.Loop() {
		_ = Uniform[float32](rng)
	}
}
