// Example E02 — Logical gates suite (AND / OR / NAND).
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E02: the same
// 2-input topology learns three different functions in turn, illustrating
// how trainable parameters specialise to whatever task the loss demands.
// XOR is intentionally excluded — it needs the larger XOR example (E01)
// because two-neuron hidden layers cannot represent it.
package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

// gate captures a named truth table over two binary inputs. Putting the
// targets in code keeps the example dependency-free and lets Go optimise
// the small slices into static data.
type gate struct {
	Name    string
	Targets []float32
}

func main() {
	gates := []gate{
		{Name: "AND", Targets: []float32{0, 0, 0, 1}},
		{Name: "OR", Targets: []float32{0, 1, 1, 1}},
		{Name: "NAND", Targets: []float32{1, 1, 1, 0}},
	}
	for _, g := range gates {
		final := trainGate(g)
		fmt.Printf("[%s] final loss = %.6f\n", g.Name, final)
	}
}

// inputs is the shared 2-bit input set used by every gate. Defined as a
// package-level slice so each call to trainGate reuses the same backing
// array — the dataset is read-only and not mutated by Fit.
var inputs = [][]float32{
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1},
}

// trainGate builds a fresh network and trains it on the supplied truth
// table. Returns the final mean-epoch loss so callers (including tests)
// can assert on the convergence threshold without scraping stdout.
func trainGate(g gate) float32 {
	n, err := nn.NewBuilder[float32]().
		Input(2).
		Dense(2, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithLearningRate(0.5).
		WithLoss(loss.MSE).
		WithMaxIterations(5_000).
		WithLossLimit(1e-4).
		Compile()
	if err != nil {
		fmt.Printf("[%s] Compile failed: %v\n", g.Name, err)
		return 1
	}

	dataset := make([]nn.Sample[float32], len(inputs))
	for i, in := range inputs {
		dataset[i] = nn.Sample[float32]{Input: in, Target: []float32{g.Targets[i]}}
	}
	_, finalLoss, err := n.Fit(dataset)
	if err != nil {
		fmt.Printf("[%s] Fit failed: %v\n", g.Name, err)
		return 1
	}

	correct := 0
	for i, in := range inputs {
		out, err := n.Query(in)
		if err != nil {
			continue
		}
		predicted := float32(0)
		if out[0] >= 0.5 {
			predicted = 1
		}
		if predicted == g.Targets[i] {
			correct++
		}
		fmt.Printf("  Query(%v) = %.4f → %.0f  (target %.0f)\n",
			in, out[0], predicted, g.Targets[i])
	}
	fmt.Printf("[%s] accuracy = %d/%d\n", g.Name, correct, len(inputs))
	return finalLoss
}
