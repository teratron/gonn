// Example E01 — XOR (Both styles).
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E01:
// the smallest non-linear classification proves that backprop wires the
// hidden layer correctly. The dataset is hard-coded; the same network
// is built twice (Builder + Functional Options) so the per-style code
// can be diffed line-for-line.
package main

import "fmt"

func main() {
	fmt.Println("== Builder API ==")
	runBuilder()
	fmt.Println()
	fmt.Println("== Functional Options API ==")
	runOptions()
}

// xorDataset returns the canonical 4-sample XOR set used by both styles.
// Splitting it out keeps the builder / options files focused on API
// shape rather than data plumbing.
func xorDataset() (inputs, targets [][]float32) {
	inputs = [][]float32{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets = [][]float32{
		{0},
		{1},
		{1},
		{0},
	}
	return inputs, targets
}
