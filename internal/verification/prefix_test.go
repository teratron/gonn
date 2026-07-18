package verification

import (
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/layer/embedding"
	"github.com/teratron/gonn/pkg/layer/recurrent"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

func ramp(n int, step float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = float64(i) * step
	}
	return out
}

// TestConv2DUnfreezes guards audit C1: Conv2D kernel weights must change under
// training now that ApplyGradSGD is wired via the capability interface.
func TestConv2DUnfreezes(t *testing.T) {
	n := nn.MustNew(
		nn.WithInput[float64](16),
		nn.WithInputShape[float64](1, 4, 4),
		nn.WithConv2D[float64](2, 1, 3, 3, 1, 1, conv.PadValid, true),
		nn.WithFlatten2D[float64](),
		nn.WithHiddenLayer[float64](6, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.5),
		nn.WithMaxIterations[float64](50),
		nn.WithLossLimit[float64](1e-9),
		nn.WithWeightInitSeed[float64](3),
	)
	c2d := n.Config().ConvPrefix[0].(*conv.Conv2D[float64])
	before := append([]float64(nil), c2d.Weights...)
	samples := []nn.Sample[float64]{
		{Input: ramp(16, 0.1), Target: []float64{1}},
		{Input: ramp(16, -0.05), Target: []float64{0}},
	}
	if _, _, err := n.Fit(samples); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if d := absDiffMax(before, c2d.Weights); d == 0 {
		t.Errorf("Conv2D kernel weights unchanged after training — still frozen")
	}
}

// TestSimpleRNNUnfreezes guards audit C2: recurrent input weights must change.
func TestSimpleRNNUnfreezes(t *testing.T) {
	n := nn.MustNew(
		nn.WithInput[float64](4),
		nn.WithSimpleRNN[float64](2, 2, 3),
		nn.WithHiddenLayer[float64](5, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.5),
		nn.WithMaxIterations[float64](50),
		nn.WithLossLimit[float64](1e-9),
		nn.WithWeightInitSeed[float64](3),
	)
	rnn := n.Config().ConvPrefix[0].(*recurrent.SimpleRNN[float64])
	before := append([]float64(nil), rnn.Wxh...)
	samples := []nn.Sample[float64]{
		{Input: []float64{0.1, 0.9, 0.4, 0.2}, Target: []float64{1}},
		{Input: []float64{0.8, 0.2, 0.6, 0.7}, Target: []float64{0}},
	}
	if _, _, err := n.Fit(samples); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if d := absDiffMax(before, rnn.Wxh); d == 0 {
		t.Errorf("SimpleRNN Wxh unchanged after training — still frozen")
	}
}

// TestEmbeddingStackCompiles guards audit C3: WithEmbeddingStack used to panic
// during compile because Forward ran before Init on nil caches.
func TestEmbeddingStackCompiles(t *testing.T) {
	n, err := nn.New(
		nn.WithInput[float64](3),
		nn.WithEmbeddingStack[float64](5, 3, 2, embedding.Learnable),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLearningRate[float64](0.05),
		nn.WithMaxIterations[float64](1),
		nn.WithWeightInitSeed[float64](9),
	)
	if err != nil {
		t.Fatalf("compile with embedding stack failed: %v", err)
	}
	if _, err := n.Query([]float64{1, 2, 3}); err != nil {
		t.Fatalf("Query: %v", err)
	}
}

// TestPositionalGradResets guards audit C4: the learnable positional gradient
// accumulator must be cleared each step (via ApplyGradSGD), so repeated
// backward passes on the same input yield the same magnitude rather than a
// linearly growing one.
func TestPositionalGradResets(t *testing.T) {
	rng, _ := utils.NewRNG(9)
	es := embedding.NewEmbeddingStack[float64](5, 3, 2, embedding.Learnable)
	es.Init(rng)
	ids := []int{1, 2, 3}
	up := ramp(6, 0.1)

	// First step: accumulate then apply+reset.
	_, _ = es.ForwardIDs(ids)
	es.Backward(up)
	g1, _ := es.Positional.GradSlots()
	first := g1[1]
	es.ApplyGradSGD(0) // lr 0: no weight change, but resets the accumulator

	// Second step from a clean accumulator must match the first, not double it.
	_, _ = es.ForwardIDs(ids)
	es.Backward(up)
	g2, _ := es.Positional.GradSlots()
	if g2[1] != first {
		t.Errorf("positional grad not reset: step1=%.4f step2=%.4f (accumulating)", first, g2[1])
	}
}
