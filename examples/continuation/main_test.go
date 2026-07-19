package main

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/nn"
)

// TestAndTrainContinuation builds the example network end-to-end and asserts
// the post-AndTrain predictions are closer to the negated targets than to
// the original XOR targets — proves the continuation actually shifted the
// learned function.
func TestAndTrainContinuation(t *testing.T) {
	net, err := nn.New(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLearningRate(0.3),
		nn.WithMaxIterations[float64](3000),
		nn.WithLossLimit(1e-3),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := net.Fit(xorDataset()); err != nil {
		t.Fatalf("Fit XOR: %v", err)
	}
	if _, _, err := net.AndTrain(negatedXOR(),
		nn.WithLearningRate(0.1),
		nn.WithMaxIterations[float64](3000),
	); err != nil {
		t.Fatalf("AndTrain: %v", err)
	}
	preds, err := predict(net)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	// Negated XOR targets are [1, 0, 0, 1].
	wantNeg := []float64{1, 0, 0, 1}
	wantXOR := []float64{0, 1, 1, 0}
	var distNeg, distXOR float64
	for i, y := range preds {
		distNeg += math.Abs(y - wantNeg[i])
		distXOR += math.Abs(y - wantXOR[i])
	}
	if distNeg >= distXOR {
		t.Errorf("post-AndTrain distance to negated %.4f not closer than to XOR %.4f", distNeg, distXOR)
	}
}
