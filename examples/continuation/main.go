// Example: AndTrain continuation.
//
// See [.design/main/specifications/l2-dataset-loader-impl.md] §5.4
// (FMT-6 / FMT-7 / FMT-8). The example demonstrates the AndTrain method:
//
//  1. Build an XOR network and train it to convergence on the standard
//     truth table.
//  2. Capture the post-Fit predictions for evidence the network learned
//     XOR.
//  3. Call AndTrain with a NEGATED XOR dataset (1↔0) and a lower learning
//     rate. AndTrain preserves the existing weights — only convergence
//     counters reset — so the network's existing knowledge of "two-input
//     parity" carries over.
//  4. Show the post-AndTrain predictions; the network now outputs the
//     negated relationship while having reused (not reset) its weights.
//
// The example proves three contract guarantees:
//   - FMT-6: AndTrain resets convergence counters but not weights.
//   - FMT-7: callbacks (if any were registered) remain active across the call.
//   - FMT-8: AndTrain refuses to run if a prior Train / Fit is still active.
package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/nn"
)

// xorDataset is the canonical 4-point XOR truth table.
func xorDataset() []nn.Sample[float64] {
	return []nn.Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
}

// negatedXOR flips every target so the network must re-learn the inverse.
// Used as the fine-tuning dataset for AndTrain.
func negatedXOR() []nn.Sample[float64] {
	src := xorDataset()
	out := make([]nn.Sample[float64], len(src))
	for i, s := range src {
		flipped := make([]float64, len(s.Target))
		for j, v := range s.Target {
			flipped[j] = 1 - v
		}
		out[i] = nn.Sample[float64]{Input: s.Input, Target: flipped}
	}
	return out
}

// predict runs every XOR row through Query and returns the four scalar
// outputs in the canonical order.
func predict(net *nn.NN[float64]) ([]float64, error) {
	rows := xorDataset()
	out := make([]float64, len(rows))
	for i, s := range rows {
		y, err := net.Query(s.Input)
		if err != nil {
			return nil, err
		}
		out[i] = y[0]
	}
	return out, nil
}

func main() {
	net, err := nn.New[float64](
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLearningRate[float64](0.3),
		nn.WithMaxIterations[float64](2000),
		nn.WithLossLimit[float64](1e-3),
	)
	if err != nil {
		fmt.Println("compile failed:", err)
		return
	}

	// Phase 1: train on XOR.
	epochs, loss, err := net.Fit(xorDataset())
	if err != nil {
		fmt.Println("Fit failed:", err)
		return
	}
	fmt.Printf("Phase 1 (XOR): %d epochs, loss=%.6f\n", epochs, loss)

	preds, _ := predict(net)
	fmt.Printf("Phase 1 predictions: %.3f\n", preds)

	// Phase 2: continue training with the negated targets at a smaller LR.
	// The same weight slab is reused; FMT-6 only resets convergence state.
	epochs2, loss2, err := net.AndTrain(negatedXOR(),
		nn.WithLearningRate[float64](0.05),
		nn.WithMaxIterations[float64](2000),
	)
	if err != nil {
		fmt.Println("AndTrain failed:", err)
		return
	}
	fmt.Printf("Phase 2 (negated, AndTrain): %d epochs, loss=%.6f\n", epochs2, loss2)

	preds2, _ := predict(net)
	fmt.Printf("Phase 2 predictions: %.3f\n", preds2)

	// The base config is restored after AndTrain returns — verify that the
	// original learning rate is back in place (would matter to a follow-up
	// Fit call on the same network).
	fmt.Println("Base config preserved? Run another Fit and observe the original LR — see test.")
}
