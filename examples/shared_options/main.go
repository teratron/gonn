// Example E14 — Shared options across multiple networks (v0.5 adapted).
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E14. The spec's
// Topology B uses two hidden layers, which v0.5 compile() rejects (the
// multi-hidden patch lands in v0.6). The pedagogical point — pass a
// shared []Option[T] slice to two distinct networks — survives by
// substituting two single-hidden widths instead. Both are XOR-shaped so
// the example trains in well under a second.
package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	common := commonOpts()

	resA := train("topology-A", append(append([]nn.Option[float32]{}, common...),
		nn.WithHiddenLayer[float32](8, activation.ReLU),
		nn.WithOutput[float32](1, activation.SIGMOID),
	))

	resB := train("topology-B", append(append([]nn.Option[float32]{}, common...),
		nn.WithHiddenLayer[float32](16, activation.ReLU),
		nn.WithOutput[float32](1, activation.SIGMOID),
	))

	fmt.Println()
	fmt.Println("Side-by-side summary")
	fmt.Printf("  topology-A  loss=%.6f  epochs=%d\n", resA.loss, resA.epochs)
	fmt.Printf("  topology-B  loss=%.6f  epochs=%d\n", resB.loss, resB.epochs)
}

// commonOpts is the slice of options reused across every architecture
// in this example. Returning a fresh slice each call avoids accidental
// mutation if a future option mutates the underlying Config[T] in a way
// that depends on call order.
func commonOpts() []nn.Option[float32] {
	return []nn.Option[float32]{
		nn.WithInput[float32](2),
		nn.WithBias[float32](true),
		nn.WithLearningRate[float32](0.3),
		nn.WithLoss[float32](loss.MSE),
		nn.WithMaxIterations[float32](10_000),
		nn.WithLossLimit[float32](1e-4),
	}
}

type result struct {
	label  string
	loss   float32
	epochs uint
}

// train builds and fits one architecture. label is purely for output
// formatting; opts is the full option set already resolved by the caller
// so this function stays oblivious to which topology it is training.
func train(label string, opts []nn.Option[float32]) result {
	n, err := nn.New(opts...)
	if err != nil {
		fmt.Printf("[%s] New failed: %v\n", label, err)
		return result{label: label, loss: 1}
	}
	dataset := []nn.Sample[float32]{
		{Input: []float32{0, 0}, Target: []float32{0}},
		{Input: []float32{0, 1}, Target: []float32{1}},
		{Input: []float32{1, 0}, Target: []float32{1}},
		{Input: []float32{1, 1}, Target: []float32{0}},
	}
	epochs, finalLoss, err := n.Fit(dataset)
	if err != nil {
		fmt.Printf("[%s] Fit failed: %v\n", label, err)
		return result{label: label, loss: 1}
	}
	fmt.Printf("[%s] epochs=%d loss=%.6f\n", label, epochs, finalLoss)
	return result{label: label, loss: finalLoss, epochs: epochs}
}
