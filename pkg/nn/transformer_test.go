package nn

import (
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer/transformer"
	"github.com/teratron/gonn/pkg/loss"
)

// newTestTrCfg returns a minimal TransformerConfig for pkg/nn integration tests.
func newTestTrCfg() transformer.TransformerConfig[float64] {
	return transformer.TransformerConfig[float64]{
		SeqLen:   4,
		Dmodel:   8,
		NumHeads: 2,
		Dff:      16,
	}
}

// TestWithTransformer_EncoderBlock verifies that WithEncoderBlock compiles and
// that the conv-prefix shape passes through compile() without error.
func TestWithTransformer_EncoderBlock(t *testing.T) {
	t.Parallel()
	cfg := newTestTrCfg()
	n, err := New(
		WithInput[float64](uint(cfg.SeqLen*cfg.Dmodel)),
		WithEncoderBlock[float64](cfg),
		WithHiddenLayer[float64](4, activation.ReLU),
		WithOutput[float64](1, activation.SIGMOID),
		WithLoss[float64](loss.MSE),
		WithLearningRate[float64](0.01),
	)
	if err != nil {
		t.Fatalf("New with WithEncoderBlock: %v", err)
	}
	if n == nil {
		t.Fatal("New returned nil")
	}
}

// TestWithTransformer_DecoderBlock verifies that WithDecoderBlock compiles.
func TestWithTransformer_DecoderBlock(t *testing.T) {
	t.Parallel()
	cfg := newTestTrCfg()
	n, err := New(
		WithInput[float64](uint(cfg.SeqLen*cfg.Dmodel)),
		WithDecoderBlock[float64](cfg),
		WithHiddenLayer[float64](4, activation.ReLU),
		WithOutput[float64](1, activation.SIGMOID),
		WithLoss[float64](loss.MSE),
		WithLearningRate[float64](0.01),
	)
	if err != nil {
		t.Fatalf("New with WithDecoderBlock: %v", err)
	}
	if n == nil {
		t.Fatal("New returned nil")
	}
}

// TestWithTransformer_EncoderStack verifies that WithEncoderStack compiles.
func TestWithTransformer_EncoderStack(t *testing.T) {
	t.Parallel()
	cfg := newTestTrCfg()
	n, err := New(
		WithInput[float64](uint(cfg.SeqLen*cfg.Dmodel)),
		WithEncoderStack[float64](cfg, 2),
		WithHiddenLayer[float64](4, activation.ReLU),
		WithOutput[float64](1, activation.SIGMOID),
		WithLoss[float64](loss.MSE),
		WithLearningRate[float64](0.01),
	)
	if err != nil {
		t.Fatalf("New with WithEncoderStack: %v", err)
	}
	if n == nil {
		t.Fatal("New returned nil")
	}
}

// TestWithTransformer_DecoderStack verifies that WithDecoderStack compiles.
func TestWithTransformer_DecoderStack(t *testing.T) {
	t.Parallel()
	cfg := newTestTrCfg()
	n, err := New(
		WithInput[float64](uint(cfg.SeqLen*cfg.Dmodel)),
		WithDecoderStack[float64](cfg, 2),
		WithHiddenLayer[float64](4, activation.ReLU),
		WithOutput[float64](1, activation.SIGMOID),
		WithLoss[float64](loss.MSE),
		WithLearningRate[float64](0.01),
	)
	if err != nil {
		t.Fatalf("New with WithDecoderStack: %v", err)
	}
	if n == nil {
		t.Fatal("New returned nil")
	}
}

// TestWithTransformer_EncoderStackFit verifies that WithEncoderStack builds a
// network whose Fit runs one epoch without error (T-18A11 primary verify criterion).
func TestWithTransformer_EncoderStackFit(t *testing.T) {
	cfg := newTestTrCfg()
	n, err := New(
		WithInput[float64](uint(cfg.SeqLen*cfg.Dmodel)),
		WithEncoderStack[float64](cfg, 2),
		WithHiddenLayer[float64](4, activation.ReLU),
		WithOutput[float64](1, activation.SIGMOID),
		WithLoss[float64](loss.MSE),
		WithLearningRate[float64](0.01),
		WithMaxIterations[float64](1),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	x := make([]float64, cfg.SeqLen*cfg.Dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.05
	}
	_, _, err = n.Fit([]Sample[float64]{{Input: x, Target: []float64{1.0}}})
	if err != nil {
		t.Fatalf("Fit one epoch: %v", err)
	}
}
