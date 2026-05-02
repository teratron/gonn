package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

// runOptions trains the same XOR network as runBuilder but via the
// Functional Options API (Style B). Both styles converge on the same
// internal Config[T] inside compile() — finalLoss should be statistically
// indistinguishable between calls aside from RNG-driven init drift.
func runOptions() float32 {
	inputs, targets := xorDataset()

	n, err := nn.New[float32](
		nn.WithInput[float32](2),
		nn.WithBias[float32](true),
		nn.WithHiddenLayer[float32](4, activation.SIGMOID),
		nn.WithOutput[float32](1, activation.SIGMOID),
		nn.WithLearningRate[float32](0.3),
		nn.WithLoss[float32](loss.MSE),
		nn.WithMaxIterations[float32](10_000),
		nn.WithLossLimit[float32](1e-4),
		nn.WithWeightInit[float32](nn.WeightInitXavier),
	)
	if err != nil {
		fmt.Printf("Options New failed: %v\n", err)
		return 1
	}

	dataset := makeSamples(inputs, targets)
	epochs, finalLoss, err := n.Fit(dataset)
	if err != nil {
		fmt.Printf("Options Fit failed: %v\n", err)
		return 1
	}
	fmt.Printf("trained for %d epochs, final mean loss = %.6f\n", epochs, finalLoss)

	for i, in := range inputs {
		out, err := n.Query(in)
		if err != nil {
			fmt.Printf("Query[%d] failed: %v\n", i, err)
			continue
		}
		fmt.Printf("  Query(%v) = %.4f  (target %.0f)\n", in, out[0], targets[i][0])
	}
	return finalLoss
}
