package quantization

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

// trainedXORNet builds and trains the canonical XOR MLP used by the
// dense-head quantization tests.
func trainedXORNet(t *testing.T) *nn.NN[float64] {
	t.Helper()
	n := nn.MustNew(
		nn.WithInput[float64](2),
		nn.WithBias[float64](true),
		nn.WithHiddenLayer[float64](6, activation.SIGMOID),
		nn.WithHiddenLayer[float64](4, activation.TanH),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.5),
		nn.WithMaxIterations[float64](3000),
		nn.WithLossLimit[float64](0.001),
		nn.WithWeightInitSeed[float64](42),
	)
	samples := []nn.Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
	if _, _, err := n.Fit(samples); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	return n
}

// TestQuantizeDenseHeadMatchesFloat guards the audit finding that MLPs were
// never quantized (QuantizedDense existed with zero callers): the quantized
// chain must reproduce the float Query within int8 tolerance on all four
// XOR corners.
func TestQuantizeDenseHeadMatchesFloat(t *testing.T) {
	n := trainedXORNet(t)
	qnet, err := Quantize(n, nil, DefaultQuantizationConfig[float64]())
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	// 2 hidden + output = 3 dense stages, no conv prefix.
	if len(qnet.Layers) != 3 {
		t.Fatalf("quantized layer count = %d, want 3 (dense head)", len(qnet.Layers))
	}
	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for _, in := range inputs {
		want, err := n.Query(in)
		if err != nil {
			t.Fatalf("Query: %v", err)
		}
		got := qnet.Forward(in)
		if len(got) != len(want) {
			t.Fatalf("Forward(%v) length %d, want %d", in, len(got), len(want))
		}
		if diff := math.Abs(got[0] - want[0]); diff > 0.05 {
			t.Errorf("Forward(%v) = %.4f, float = %.4f, |diff| = %.4f > 0.05", in, got[0], want[0], diff)
		}
	}
}

// TestQuantizeDenseHeadQNNRoundTrip: the dense head (with activations) must
// survive Save/Load bit-exact.
func TestQuantizeDenseHeadQNNRoundTrip(t *testing.T) {
	n := trainedXORNet(t)
	qnet, err := Quantize(n, nil, DefaultQuantizationConfig[float64]())
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	path := filepath.Join(t.TempDir(), "mlp.qnn.json")
	if err := qnet.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	qnet2, err := Load[float64](path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for _, in := range [][]float64{{0, 1}, {1, 1}} {
		y1 := qnet.Forward(in)
		y2 := qnet2.Forward(in)
		if y1[0] != y2[0] {
			t.Errorf("Forward(%v): before=%.12f after=%.12f (activation not persisted?)", in, y1[0], y2[0])
		}
	}
}

// TestBaselineHashSeesWeights guards the audit finding that baselineHash
// digested "{}" cell bundles: two same-shape networks with different weights
// must hash differently.
func TestBaselineHashSeesWeights(t *testing.T) {
	mk := func(seed uint64) *nn.NN[float64] {
		return nn.MustNew(
			nn.WithInput[float64](2),
			nn.WithHiddenLayer[float64](4, activation.SIGMOID),
			nn.WithOutput[float64](1, activation.SIGMOID),
			nn.WithWeightInitSeed[float64](seed),
		)
	}
	h1, err := baselineHash(mk(1))
	if err != nil {
		t.Fatalf("hash1: %v", err)
	}
	h2, err := baselineHash(mk(2))
	if err != nil {
		t.Fatalf("hash2: %v", err)
	}
	if h1 == h2 {
		t.Error("baselineHash identical for different weights — hash does not cover the trained state")
	}
	h1b, _ := baselineHash(mk(1))
	if h1 != h1b {
		t.Error("baselineHash not deterministic for identical networks")
	}
}
