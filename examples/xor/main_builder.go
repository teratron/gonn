package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

// runBuilder trains an XOR network using the Builder API (Style A) and
// returns the final mean-epoch loss so smoke tests can assert on the
// number without re-parsing stdout.
//
// Topology and hyperparameters are the canonical XOR setup used as a
// reference implementation across the other XOR-shaped examples.
func runBuilder() float32 {
	inputs, targets := xorDataset()

	n, err := nn.NewBuilder[float32]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithLearningRate(0.3).
		WithLoss(loss.MSE).
		WithMaxIterations(10_000).
		WithLossLimit(1e-4).
		WithWeightInit(nn.WeightInitXavier).
		Compile()
	if err != nil {
		fmt.Printf("Builder Compile failed: %v\n", err)
		return 1
	}

	dataset := makeSamples(inputs, targets)
	epochs, finalLoss, err := n.Fit(dataset)
	if err != nil {
		fmt.Printf("Builder Fit failed: %v\n", err)
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

// makeSamples bundles parallel input / target slices into the
// pkg/nn.Sample[T] form expected by Fit. Lives in the builder file
// because it is the first file in the example to need it; the options
// file imports the same helper from the package.
func makeSamples(inputs, targets [][]float32) []nn.Sample[float32] {
	out := make([]nn.Sample[float32], len(inputs))
	for i := range inputs {
		out[i] = nn.Sample[float32]{Input: inputs[i], Target: targets[i]}
	}
	return out
}
