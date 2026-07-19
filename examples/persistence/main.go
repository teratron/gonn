// Save / reload (round-trip integrity).
//
// Trains the canonical XOR network, writes its config + weights to disk
// via nn.Save, reconstructs the network with nn.Load, and asserts the
// post-reload Query output matches the original within float-32 tolerance.
//
// The default run uses XOR (single hidden); the smoke test in main_test.go
// exercises a 2-hidden round-trip via the same code path.
package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

// hiddenSpec mirrors one hidden layer entry locally so the topology is a
// single source of truth for the builder chain.
type hiddenSpec struct {
	size uint
	act  activation.Type
	bias bool
}

// outputSpec is the matching record for the Output layer.
type outputSpec struct {
	size uint
	act  activation.Type
	bias bool
}

// trainConfig groups every hyperparameter the example needs to wire a
// network. The fields are exposed so tests can override individual
// parts without rebuilding the whole literal.
type trainConfig struct {
	inputSize uint
	hidden    []hiddenSpec
	output    outputSpec
	loss      loss.Type
	rate      float32
	maxIters  uint
	lossLimit float32
}

// xorTopology returns the canonical single-hidden XOR setup used by
// main(). Single-hidden remains the regression baseline for the
// example; multi-hidden coverage lives in main_test.go.
func xorTopology() trainConfig {
	return trainConfig{
		inputSize: 2,
		hidden: []hiddenSpec{
			{size: 4, act: activation.SIGMOID, bias: true},
		},
		output:    outputSpec{size: 1, act: activation.SIGMOID, bias: true},
		loss:      loss.MSE,
		rate:      0.3,
		maxIters:  10_000,
		lossLimit: 1e-4,
	}
}

func main() {
	if err := run("nn-roundtrip", xorTopology()); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// xorDataset is duplicated across examples on purpose — keeping the
// dataset literal next to its consumer makes the example self-contained.
func xorDataset() []nn.Sample[float32] {
	return []nn.Sample[float32]{
		{Input: []float32{0, 0}, Target: []float32{0}},
		{Input: []float32{0, 1}, Target: []float32{1}},
		{Input: []float32{1, 0}, Target: []float32{1}},
		{Input: []float32{1, 1}, Target: []float32{0}},
	}
}

// run wires the round-trip end-to-end: train → Save → Load → compare.
// Tests call it directly so the assertion can read the numeric drift
// instead of scraping stdout.
func run(label string, tc trainConfig) error {
	original, err := train(tc)
	if err != nil {
		return fmt.Errorf("train: %w", err)
	}

	originalOutputs, err := queryAll(original)
	if err != nil {
		return fmt.Errorf("query original: %w", err)
	}
	fmt.Println("Pre-save outputs:")
	printOutputs(originalOutputs)

	dir, err := os.MkdirTemp("", label+"-*")
	if err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	defer os.RemoveAll(dir)

	cfgPath := filepath.Join(dir, "config.json")
	weightsPath := filepath.Join(dir, "weights.json")

	if err := original.Save(cfgPath, weightsPath); err != nil {
		return fmt.Errorf("save: %w", err)
	}
	fmt.Printf("\nArtefacts written to %s\n", dir)

	reloaded, err := nn.Load[float32](cfgPath, weightsPath)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}

	reloadedOutputs, err := queryAll(reloaded)
	if err != nil {
		return fmt.Errorf("query reloaded: %w", err)
	}
	fmt.Println("\nPost-reload outputs:")
	printOutputs(reloadedOutputs)

	maxDrift := float32(0)
	for i := range originalOutputs {
		for j := range originalOutputs[i] {
			d := absDiff(originalOutputs[i][j], reloadedOutputs[i][j])
			if d > maxDrift {
				maxDrift = d
			}
		}
	}
	fmt.Printf("\nmax |original - reloaded| = %.3e\n", maxDrift)
	return nil
}

// train builds the network described by tc and runs Fit on the XOR
// dataset. Returns the trained NN; callers persist it via Save.
func train(tc trainConfig) (*nn.NN[float32], error) {
	b := nn.NewBuilder[float32]().Input(tc.inputSize)
	for _, h := range tc.hidden {
		b = b.Dense(h.size, h.act, h.bias)
	}
	n, err := b.
		Output(tc.output.size, tc.output.act, tc.output.bias).
		WithLearningRate(tc.rate).
		WithLoss(tc.loss).
		WithMaxIterations(tc.maxIters).
		WithLossLimit(tc.lossLimit).
		Compile()
	if err != nil {
		return nil, err
	}
	if _, _, err := n.Fit(xorDataset()); err != nil {
		return nil, err
	}
	return n, nil
}

func queryAll(n *nn.NN[float32]) ([][]float32, error) {
	inputs := xorDataset()
	out := make([][]float32, len(inputs))
	for i, s := range inputs {
		v, err := n.Query(s.Input)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func printOutputs(outputs [][]float32) {
	inputs := xorDataset()
	for i, s := range inputs {
		fmt.Printf("  Query(%v) = %.6f\n", s.Input, outputs[i][0])
	}
}

func absDiff(a, b float32) float32 {
	return float32(math.Abs(float64(a - b)))
}
