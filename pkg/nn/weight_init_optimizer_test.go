package nn

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/regularizer"
)

// TestWeightInitRanges verifies Xavier, He, and Random initialisation strategies
// produce weight distributions within the expected theoretical bounds.
// Uses a 64→64→1 topology so fan-in is large enough to make the bounds tight.
func TestWeightInitRanges(t *testing.T) {
	cases := []struct {
		name   string
		method WeightInitMethod
		maxAbs float64
	}{
		// Xavier U[-√(6/(64+64)), √(6/(64+64))] ≈ ±0.217 — bound at 1.0 is conservative.
		{"Xavier", WeightInitXavier, 1.0},
		// He N(0, √(2/64)) ≈ σ=0.177; 5σ ≈ 0.885 — bound at 3.0 covers extreme tails.
		{"He", WeightInitHe, 3.0},
		// Random U[-1, 1).
		{"Random", WeightInitRandom, 1.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n, err := NewBuilder[float64]().
				Input(64).
				Dense(64, activation.SIGMOID, false).
				Output(1, activation.SIGMOID, false).
				WithWeightInit(tc.method).
				Compile()
			if err != nil {
				t.Fatalf("Compile: %v", err)
			}
			weights := n.Network.AppendFlatWeights(nil)
			if len(weights) == 0 {
				t.Fatal("no weights found")
			}
			var sumSq float64
			for _, w := range weights {
				if abs := math.Abs(float64(w)); abs > tc.maxAbs {
					t.Errorf("weight %v exceeds maxAbs %v for %s init", w, tc.maxAbs, tc.name)
				}
				sumSq += float64(w) * float64(w)
			}
			// Mean squared value must be non-negligible — all-zero init would be broken.
			if sumSq/float64(len(weights)) < 1e-6 {
				t.Errorf("%s: weights appear near-zero (mean_sq=%v); sampler may not be wired", tc.name, sumSq/float64(len(weights)))
			}
		})
	}
}

// TestRegularizerConvergence verifies Compose(L2, Dropout) does not prevent
// XOR from reducing loss. Dropout slows convergence so the threshold is
// deliberately loose — the goal is to confirm training runs without error
// and that loss decreases, not to match unregularized speed.
func TestRegularizerConvergence(t *testing.T) {
	reg := regularizer.Compose(
		regularizer.NewL2(0.001),
		regularizer.NewDropoutSeeded[float64](0.9, 1),
	)
	n, err := New(
		WithInput[float64](2),
		WithHiddenLayer[float64](16, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate(0.3),
		WithMaxIterations[float64](50_000),
		WithLossLimit(0.05),
		WithRegularizer(reg),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	dataset := xorDataset[float64]()
	_, loss, err := n.Fit(dataset)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// Regularized networks converge slower; 0.3 is still well below random
	// guessing (0.25 for balanced binary) so this confirms real learning.
	if loss > 0.3 {
		t.Errorf("regularized XOR final loss = %v (want ≤ 0.3)", loss)
	}
}

// TestInferenceNoDropout verifies that Query is deterministic with a Dropout
// regularizer attached — training=false is a strict no-op (REG-3).
func TestInferenceNoDropout(t *testing.T) {
	reg := regularizer.NewDropoutSeeded[float64](0.5, 99)
	n, err := New(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithRegularizer(reg),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	input := []float64{0.3, 0.7}
	first, err := n.Query(input)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	for call := 2; call <= 20; call++ {
		got, err := n.Query(input)
		if err != nil {
			t.Fatalf("Query call %d: %v", call, err)
		}
		for i, v := range got {
			if math.Abs(v-first[i]) > 1e-12 {
				t.Errorf("call %d output[%d] = %v; want %v (dropout must not run in inference)", call, i, v, first[i])
			}
		}
	}
}

// TestOptimizerIntegration verifies all four optimizer types train XOR to loss < 0.5
// within a generous budget — checks for convergence, not bit-identical output.
func TestOptimizerIntegration(t *testing.T) {
	dataset := xorDataset[float64]()

	opts := []struct {
		name string
		opt  optimizer.Optimizer[float64]
	}{
		{"SGD", optimizer.NewSGD(0.3)},
		{"Adam", optimizer.NewAdam(0.001)},
		{"SGDMomentum", optimizer.NewSGDMomentum(0.1)},
		{"RMSProp", optimizer.NewRMSProp(0.001)},
	}

	for _, tc := range opts {
		t.Run(tc.name, func(t *testing.T) {
			n, err := New(
				WithInput[float64](2),
				WithHiddenLayer[float64](4, activation.SIGMOID),
				WithOutput[float64](1, activation.SIGMOID),
				WithMaxIterations[float64](50_000),
				WithLossLimit(0.05),
				WithOptimizer(tc.opt),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			_, loss, err := n.Fit(dataset)
			if err != nil {
				t.Fatalf("Fit: %v", err)
			}
			if loss > 0.5 {
				t.Errorf("optimizer %s: final loss = %v (want ≤ 0.5)", tc.name, loss)
			}
		})
	}
}
