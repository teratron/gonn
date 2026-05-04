// Example E09 — Save / reload (round-trip integrity).
//
// See [.design/specifications/l2-usage-examples.md] §5.2 / E09 — ungated
// after Phase 3 promoted l1-network-persistence to Stable. This example
// trains the canonical XOR network, writes its config + weights to disk
// via pkg/persistence, builds a fresh network, copies the loaded weights
// into it, and asserts the post-reload Query output matches the original
// within float-32 tolerance (PERS-4).
//
// The example also documents the v0.5 conversion seam — pkg/nn does not
// yet expose dump / load hooks, so the bundle accessors on
// network.Network[T] are walked manually. Future versions of the facade
// are expected to fold this glue into nn.Save / nn.Load.
package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/persistence"
)

const (
	hiddenSize = 4
	hiddenAct  = activation.SIGMOID
	outputAct  = activation.SIGMOID
	lossMode   = loss.MSE
	rate       = 0.3
	maxIters   = 10_000
	lossLimit  = 1e-4
)

func main() {
	if err := run("nn-roundtrip"); err != nil {
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

// run wires the round-trip end-to-end and returns the maximum absolute
// difference between original and reloaded Query outputs. Tests call it
// directly so the assertion can read the numeric drift instead of
// scraping stdout.
func run(label string) error {
	original, err := train()
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

	cfg := buildConfigDoc()
	weights := extractWeights(original)
	if err := persistence.WriteConfig(cfgPath, cfg); err != nil {
		return fmt.Errorf("write config: %w", err)
	}
	if err := persistence.WriteWeights(weightsPath, cfg, weights); err != nil {
		return fmt.Errorf("write weights: %w", err)
	}
	fmt.Printf("\nArtefacts written to %s\n", dir)

	loadedCfg, loadedWeights, err := persistence.ReadWeights[float32](cfgPath, weightsPath)
	if err != nil {
		return fmt.Errorf("read weights: %w", err)
	}
	if loadedCfg.InputSize != cfg.InputSize {
		return fmt.Errorf("InputSize drift: want %d got %d", cfg.InputSize, loadedCfg.InputSize)
	}

	rebuilt, err := buildBlank()
	if err != nil {
		return fmt.Errorf("blank net: %w", err)
	}
	if err := installWeights(rebuilt, loadedWeights); err != nil {
		return fmt.Errorf("install weights: %w", err)
	}

	rebuiltOutputs, err := queryAll(rebuilt)
	if err != nil {
		return fmt.Errorf("query rebuilt: %w", err)
	}
	fmt.Println("\nPost-reload outputs:")
	printOutputs(rebuiltOutputs)

	maxDrift := float32(0)
	for i := range originalOutputs {
		for j := range originalOutputs[i] {
			d := absDiff(originalOutputs[i][j], rebuiltOutputs[i][j])
			if d > maxDrift {
				maxDrift = d
			}
		}
	}
	fmt.Printf("\nmax |original - reloaded| = %.3e\n", maxDrift)
	return nil
}

func train() (*nn.NN[float32], error) {
	n, err := nn.NewBuilder[float32]().
		Input(2).
		Dense(hiddenSize, hiddenAct, true).
		Output(1, outputAct, true).
		WithLearningRate(rate).
		WithLoss(lossMode).
		WithMaxIterations(maxIters).
		WithLossLimit(lossLimit).
		Compile()
	if err != nil {
		return nil, err
	}
	if _, _, err := n.Fit(xorDataset()); err != nil {
		return nil, err
	}
	return n, nil
}

// buildBlank constructs a network with the same topology as train() but
// without running Fit. Random init guarantees its outputs differ from
// the trained network — installWeights then overwrites those weights.
func buildBlank() (*nn.NN[float32], error) {
	return nn.NewBuilder[float32]().
		Input(2).
		Dense(hiddenSize, hiddenAct, true).
		Output(1, outputAct, true).
		WithLearningRate(rate).
		WithLoss(lossMode).
		WithMaxIterations(1).
		Compile()
}

// buildConfigDoc returns the on-disk projection of the topology used in
// train(). Kept alongside train() so the two cannot drift — any change
// to the architecture must update both call sites.
func buildConfigDoc() persistence.ConfigDoc[float32] {
	return persistence.ConfigDoc[float32]{
		LibVersion: "0.1.0",
		InputSize:  2,
		HiddenLayers: []persistence.HiddenLayerDoc{
			{Size: hiddenSize, Activation: hiddenAct.String(), Bias: true},
		},
		Output: persistence.OutputDoc{Size: 1, Activation: outputAct.String(), Bias: true},
		Training: persistence.TrainingDoc[float32]{
			LearningRate:  rate,
			Loss:          lossMode.String(),
			LossLimit:     lossLimit,
			MaxIterations: maxIters,
			WeightInit:    "xavier",
		},
	}
}

// extractWeights walks a trained network's bundles and packages each
// layer's axon weights + bias contributions into the on-disk schema.
// Hidden cells have len(Input)+1 axons (last one is bias); output cells
// have len(Hidden)+1 axons (last one is bias).
//
// Track C (Phase 5 v0.6) generalises this helper to walk every entry in
// n.Network.Hiddens; for v0.5 single-hidden examples we still emit one
// "hidden_0" layer plus output.
func extractWeights(n *nn.NN[float32]) persistence.WeightsDoc[float32] {
	hiddenCells := n.Network.Hiddens[0].Cells()
	hiddenLayer := persistence.LayerWeights[float32]{
		Name:    "hidden_0",
		Weights: make([][]float32, len(hiddenCells)),
		Biases:  make([]float32, len(hiddenCells)),
	}
	for i, h := range hiddenCells {
		row := make([]float32, len(h.Axons)-1) // last axon is bias
		for j := 0; j < len(h.Axons)-1; j++ {
			row[j] = h.Axons[j].Weight
		}
		hiddenLayer.Weights[i] = row
		hiddenLayer.Biases[i] = h.Axons[len(h.Axons)-1].Weight
	}

	outputCells := n.Network.Output.Cells()
	outputLayer := persistence.LayerWeights[float32]{
		Name:    "output",
		Weights: make([][]float32, len(outputCells)),
		Biases:  make([]float32, len(outputCells)),
	}
	for i, o := range outputCells {
		row := make([]float32, len(o.Axons)-1)
		for j := 0; j < len(o.Axons)-1; j++ {
			row[j] = o.Axons[j].Weight
		}
		outputLayer.Weights[i] = row
		outputLayer.Biases[i] = o.Axons[len(o.Axons)-1].Weight
	}

	return persistence.WeightsDoc[float32]{
		Layers: []persistence.LayerWeights[float32]{hiddenLayer, outputLayer},
	}
}

// installWeights performs the inverse of extractWeights: copy the
// loaded weight values back into a freshly compiled network's axons.
// Bias axons are always the last entry in each cell's Axons slice.
func installWeights(n *nn.NN[float32], doc persistence.WeightsDoc[float32]) error {
	if len(doc.Layers) != 2 {
		return fmt.Errorf("expected 2 layers, got %d", len(doc.Layers))
	}
	hiddenCells := n.Network.Hiddens[0].Cells()
	for i, h := range hiddenCells {
		row := doc.Layers[0].Weights[i]
		if len(row) != len(h.Axons)-1 {
			return fmt.Errorf("hidden[%d] axon count mismatch: doc %d, net %d",
				i, len(row), len(h.Axons)-1)
		}
		for j, w := range row {
			h.Axons[j].Weight = w
		}
		h.Axons[len(h.Axons)-1].Weight = doc.Layers[0].Biases[i]
	}
	outputCells := n.Network.Output.Cells()
	for i, o := range outputCells {
		row := doc.Layers[1].Weights[i]
		if len(row) != len(o.Axons)-1 {
			return fmt.Errorf("output[%d] axon count mismatch: doc %d, net %d",
				i, len(row), len(o.Axons)-1)
		}
		for j, w := range row {
			o.Axons[j].Weight = w
		}
		o.Axons[len(o.Axons)-1].Weight = doc.Layers[1].Biases[i]
	}
	return nil
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
