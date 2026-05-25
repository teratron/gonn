// Example E03 — Perceptron (legacy continuity).
//
// Restored to the spec-canonical 4-hidden topology after Phase 5 lifted
// the v0.1 single-hidden gate per [l2-multihidden-impl] §5.5:
//
//	3 → Sigmoid(5) → ReLU(10) → Sigmoid(5) → Linear(2)
//
// Mixed bias per spec — first three hidden layers carry bias, the
// pre-output Sigmoid layer does not. Loss = ARCTAN, rate = 0.3, max
// iterations 100000, loss limit 1e-6, weight init Xavier (Random init
// on a 4-deep stack triggers the deep-stack soft warning, which is
// covered by pkg/nn/multihidden_test.go separately).
//
// The example uses the Builder API to mirror the [l2-usage-examples]
// §5.2 / E03 reference walk-through. The published reference query
// `[-0.52, 0.66, 0.81] → ≈ [-0.13, 0.2]` is asserted in main_test.go
// with a loose tolerance — random init drift makes ULP-precision
// comparison meaningless on this topology.
package main

import (
	"fmt"
	"time"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

// dataSet is the legacy E03 stream — eleven float32 values fed
// through a sliding window of three inputs and two targets.
func dataSet() []float32 {
	return []float32{.27, -.31, -.52, .66, .81, -.13, .2, .49, .11, -.73, .28}
}

const (
	lenInput  = 3
	lenOutput = 2
)

// build assembles the canonical 4-hidden perceptron and Compile-s it.
// Pulled out of main() so main_test.go can exercise the same wiring.
func build() (*nn.NN[float32], error) {
	return nn.NewBuilder[float32]().
		Input(lenInput).
		Dense(5, activation.SIGMOID, true).
		Dense(10, activation.ReLU, true).
		Dense(5, activation.SIGMOID, false).
		Output(lenOutput, activation.Linear, true).
		WithLoss(loss.ARCTAN).
		WithLearningRate(0.3).
		WithMaxIterations(100_000).
		WithLossLimit(1e-6).
		WithWeightInit(nn.WeightInitXavier).
		Compile()
}

// trainPerceptron streams a sliding window over dataSet() and runs
// per-sample n.Train. Returns the final query for the published
// reference window so callers can sanity-check the trained net.
func trainPerceptron(n *nn.NN[float32]) ([]float32, error) {
	data := dataSet()
	lenData := len(data) - lenOutput
	for epoch := 1; epoch <= 100_000; epoch++ {
		for i := lenInput; i <= lenData; i++ {
			if _, err := n.Train(data[i-lenInput:i], data[i:i+lenOutput]); err != nil {
				return nil, err
			}
		}
	}
	return n.Query([]float32{-.52, .66, .81})
}

func main() {
	n, err := build()
	if err != nil {
		fmt.Printf("Compile error: %v\n", err)
		return
	}

	start := time.Now()
	pred, err := trainPerceptron(n)
	if err != nil {
		fmt.Printf("Train error: %v\n", err)
		return
	}
	fmt.Printf("Elapsed: %v\n", time.Since(start))
	fmt.Printf("Query([-0.52, 0.66, 0.81]) = %v (reference ≈ [-0.13, 0.2])\n", pred)
}
