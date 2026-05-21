// Example E12 — Three styles, identical network.
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E12: a side-by-side
// proof that the Builder API, the Functional Options API, and the Preset bundle
// converge on the same Network[T] for the canonical XOR topology. The catalog
// spec calls for the three final losses to land within 1e-3 of each other —
// the smoke test asserts a looser bound that still catches drift.
package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	lossB := buildBuilder()
	lossO := buildOptions()
	lossP := buildPreset()

	fmt.Println()
	fmt.Println("Final per-style losses:")
	fmt.Printf("  Builder = %.6f\n", lossB)
	fmt.Printf("  Options = %.6f\n", lossO)
	fmt.Printf("  Preset  = %.6f\n", lossP)
}

// xorDataset reuses the canonical four-sample XOR set. Kept local to the
// example package so the file compiles standalone — no cross-example
// helper imports.
func xorDataset() []nn.Sample[float32] {
	return []nn.Sample[float32]{
		{Input: []float32{0, 0}, Target: []float32{0}},
		{Input: []float32{0, 1}, Target: []float32{1}},
		{Input: []float32{1, 0}, Target: []float32{1}},
		{Input: []float32{1, 1}, Target: []float32{0}},
	}
}

// buildBuilder constructs the XOR net via the fluent Builder API and
// returns the Fit() final loss so the smoke test can assert parity.
func buildBuilder() float32 {
	n, err := nn.NewBuilder[float32]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithLearningRate(0.3).
		WithLoss(loss.MSE).
		WithMaxIterations(10_000).
		WithLossLimit(1e-4).
		Compile()
	if err != nil {
		fmt.Printf("Builder Compile: %v\n", err)
		return 1
	}
	_, l, _ := n.Fit(xorDataset())
	fmt.Printf("Builder loss = %.6f\n", l)
	return l
}

// buildOptions builds the same network through Functional Options. The
// option list mirrors the Builder chain field-for-field so a reviewer
// can diff the two by line count.
func buildOptions() float32 {
	n, err := nn.New(
		nn.WithInput[float32](2),
		nn.WithBias[float32](true),
		nn.WithHiddenLayer[float32](4, activation.SIGMOID),
		nn.WithOutput[float32](1, activation.SIGMOID),
		nn.WithLearningRate[float32](0.3),
		nn.WithLoss[float32](loss.MSE),
		nn.WithMaxIterations[float32](10_000),
		nn.WithLossLimit[float32](1e-4),
	)
	if err != nil {
		fmt.Printf("Options New: %v\n", err)
		return 1
	}
	_, l, _ := n.Fit(xorDataset())
	fmt.Printf("Options loss = %.6f\n", l)
	return l
}

// buildPreset uses PresetXOR — the catalog's named bundle for this
// network — and demonstrates how presets compose with extra options
// (WithMaxIterations / WithLossLimit override the preset defaults).
func buildPreset() float32 {
	n, err := nn.New(
		nn.PresetXOR[float32](),
		nn.WithMaxIterations[float32](10_000),
		nn.WithLossLimit[float32](1e-4),
	)
	if err != nil {
		fmt.Printf("Preset New: %v\n", err)
		return 1
	}
	_, l, _ := n.Fit(xorDataset())
	fmt.Printf("Preset loss  = %.6f\n", l)
	return l
}
