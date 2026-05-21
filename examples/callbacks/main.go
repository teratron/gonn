// Example E11 — Progress callbacks.
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E11: trains the
// canonical XOR network with both WithEpochCallback and WithBatchCallback
// wired up. The epoch hook prints a heartbeat every 100 iterations; the
// batch hook is throttled to one line at the start of each new epoch
// stripe so stdout stays readable on small datasets.
package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	finalLoss := train()
	fmt.Printf("\nFinal loss = %.6f\n", finalLoss)
}

// xorDataset returns the canonical XOR set so the example trains
// quickly enough to make per-epoch callback output visible.
func xorDataset() []nn.Sample[float32] {
	return []nn.Sample[float32]{
		{Input: []float32{0, 0}, Target: []float32{0}},
		{Input: []float32{0, 1}, Target: []float32{1}},
		{Input: []float32{1, 0}, Target: []float32{1}},
		{Input: []float32{1, 1}, Target: []float32{0}},
	}
}

// train builds the network and returns the final mean-epoch loss after
// Fit. The body keeps callback wiring near the option list so a reader
// can see the relationship at a glance.
func train() float32 {
	var (
		epochsLogged int
		batchLogged  int
	)
	n, err := nn.New(
		nn.PresetXOR[float32](),
		nn.WithMaxIterations[float32](2_000),
		nn.WithLossLimit[float32](1e-4),
		nn.WithEpochCallback(func(epoch uint, loss float32) {
			if epoch%100 == 0 {
				epochsLogged++
				fmt.Printf("epoch=%4d  loss=%.6f\n", epoch, loss)
			}
		}),
		nn.WithBatchCallback(func(batch uint, loss float32) {
			// One log per "first batch of a new epoch stripe" — keeps stdout
			// useful without flooding it with N×4_000 lines.
			if batch == 0 {
				batchLogged++
			}
		}),
	)
	if err != nil {
		fmt.Printf("New failed: %v\n", err)
		return 1
	}
	_, l, err := n.Fit(xorDataset())
	if err != nil {
		fmt.Printf("Fit failed: %v\n", err)
		return 1
	}
	fmt.Printf("\nepoch callbacks observed: %d\nbatch first-of-epoch callbacks observed: %d\n",
		epochsLogged, batchLogged)
	return l
}
