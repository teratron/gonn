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
//
// Phase 5 / Track C generalises the extract / install helpers across the
// Hiddens slice. The default run still uses XOR (single hidden, the v0.1
// regression baseline); the smoke test in main_test.go exercises a
// 2-hidden round-trip via the same code path so the multi-hidden seam
// is covered.
package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/persistence"
)

// hiddenSpec mirrors one HiddenLayerDoc entry locally. The example
// keeps a single source of topology truth so the network builder, the
// config-doc emitter, and the extract / install helpers cannot drift.
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

// run wires the round-trip end-to-end and returns the maximum absolute
// difference between original and reloaded Query outputs. Tests call it
// directly so the assertion can read the numeric drift instead of
// scraping stdout.
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

	cfg := buildConfigDoc(tc)
	weights := extractWeights(original, tc)
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

	rebuilt, err := buildBlank(tc)
	if err != nil {
		return fmt.Errorf("blank net: %w", err)
	}
	if err := installWeights(rebuilt, tc, loadedWeights); err != nil {
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

// train builds the network described by tc and runs Fit on the XOR
// dataset. Returns the trained NN; callers extract weights from it.
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

// buildBlank constructs a network with the same topology as train() but
// without running Fit. Random init guarantees its outputs differ from
// the trained network — installWeights then overwrites those weights.
func buildBlank(tc trainConfig) (*nn.NN[float32], error) {
	b := nn.NewBuilder[float32]().Input(tc.inputSize)
	for _, h := range tc.hidden {
		b = b.Dense(h.size, h.act, h.bias)
	}
	return b.
		Output(tc.output.size, tc.output.act, tc.output.bias).
		WithLearningRate(tc.rate).
		WithLoss(tc.loss).
		WithMaxIterations(1).
		Compile()
}

// buildConfigDoc returns the on-disk projection of tc. Kept alongside
// train() so the two cannot drift — any change to the architecture
// must update both call sites.
func buildConfigDoc(tc trainConfig) persistence.ConfigDoc[float32] {
	hiddens := make([]persistence.HiddenLayerDoc, len(tc.hidden))
	for i, h := range tc.hidden {
		hiddens[i] = persistence.HiddenLayerDoc{
			Size:       h.size,
			Activation: h.act.String(),
			Bias:       h.bias,
		}
	}
	return persistence.ConfigDoc[float32]{
		LibVersion:   "0.2.0",
		InputSize:    tc.inputSize,
		HiddenLayers: hiddens,
		Output: persistence.OutputDoc{
			Size:       tc.output.size,
			Activation: tc.output.act.String(),
			Bias:       tc.output.bias,
		},
		Training: persistence.TrainingDoc[float32]{
			LearningRate:  tc.rate,
			Loss:          tc.loss.String(),
			LossLimit:     tc.lossLimit,
			MaxIterations: tc.maxIters,
			WeightInit:    "xavier",
		},
	}
}

// extractWeights walks every Hiddens[i] bundle plus the Output bundle
// and packages their axon weights into the on-disk schema. Per-layer
// Bias state is read from tc — when bias is false the cell carries no
// bias axon, so the row width and Biases slice shape change accordingly.
//
// Layer naming convention (matches l2-multihidden-impl §5.4): hidden_0,
// hidden_1, …, output.
func extractWeights(n *nn.NN[float32], tc trainConfig) persistence.WeightsDoc[float32] {
	layers := make([]persistence.LayerWeights[float32], 0, len(tc.hidden)+1)
	for i, h := range tc.hidden {
		cells := n.Network.Hiddens[i].Cells()
		layers = append(layers, extractLayer(
			fmt.Sprintf("hidden_%d", i),
			h.bias,
			len(cells),
			func(cellIdx int) []float32 { return axonWeights(n.Network.Hiddens[i].Cells()[cellIdx].Axons) },
		))
	}
	outCells := n.Network.Output.Cells()
	layers = append(layers, extractLayer(
		"output",
		tc.output.bias,
		len(outCells),
		func(cellIdx int) []float32 { return axonWeights(outCells[cellIdx].Axons) },
	))
	return persistence.WeightsDoc[float32]{Layers: layers}
}

// extractLayer is the shared shape-aware packager. Bias axons live at
// the tail of the cell's Axons slice when hasBias is true; otherwise
// every axon contributes to the Weights matrix.
func extractLayer(name string, hasBias bool, numCells int, axonsForCell func(int) []float32) persistence.LayerWeights[float32] {
	out := persistence.LayerWeights[float32]{
		Name:    name,
		Weights: make([][]float32, numCells),
	}
	if hasBias {
		out.Biases = make([]float32, numCells)
	}
	for i := range numCells {
		ws := axonsForCell(i)
		split := len(ws)
		if hasBias {
			split--
		}
		row := make([]float32, split)
		copy(row, ws[:split])
		out.Weights[i] = row
		if hasBias {
			out.Biases[i] = ws[split]
		}
	}
	return out
}

// installWeights performs the inverse of extractWeights: copy the
// loaded weight values back into a freshly compiled network's axons.
// Bias axons are at the tail of each cell's Axons slice when the layer
// declares bias == true.
func installWeights(n *nn.NN[float32], tc trainConfig, doc persistence.WeightsDoc[float32]) error {
	wantLayers := len(tc.hidden) + 1
	if len(doc.Layers) != wantLayers {
		return fmt.Errorf("expected %d layers (hidden chain + output), got %d", wantLayers, len(doc.Layers))
	}
	for i, h := range tc.hidden {
		cells := n.Network.Hiddens[i].Cells()
		layer := doc.Layers[i]
		if err := installLayer(fmt.Sprintf("hidden_%d", i), h.bias, len(cells), layer,
			func(cellIdx int, axonIdx int, w float32) {
				cells[cellIdx].Axons[axonIdx].Weight = w
			},
			func(cellIdx int) int { return len(cells[cellIdx].Axons) },
		); err != nil {
			return err
		}
	}
	outCells := n.Network.Output.Cells()
	outLayer := doc.Layers[len(doc.Layers)-1]
	return installLayer("output", tc.output.bias, len(outCells), outLayer,
		func(cellIdx int, axonIdx int, w float32) {
			outCells[cellIdx].Axons[axonIdx].Weight = w
		},
		func(cellIdx int) int { return len(outCells[cellIdx].Axons) },
	)
}

// installLayer mirrors extractLayer for the install side. setAxon /
// axonCount close over the live cell slice so this helper stays free
// of the generic *cell.Hidden / *cell.Output type split.
func installLayer(
	name string,
	hasBias bool,
	numCells int,
	layer persistence.LayerWeights[float32],
	setAxon func(cellIdx, axonIdx int, w float32),
	axonCount func(cellIdx int) int,
) error {
	if len(layer.Weights) != numCells {
		return fmt.Errorf("%s: doc has %d cells, net has %d", name, len(layer.Weights), numCells)
	}
	if hasBias && len(layer.Biases) != numCells {
		return fmt.Errorf("%s: bias count mismatch — doc %d, net %d", name, len(layer.Biases), numCells)
	}
	for i := range numCells {
		row := layer.Weights[i]
		expectedSplit := axonCount(i)
		if hasBias {
			expectedSplit--
		}
		if len(row) != expectedSplit {
			return fmt.Errorf("%s[%d]: weight row width %d != expected %d", name, i, len(row), expectedSplit)
		}
		for j, w := range row {
			setAxon(i, j, w)
		}
		if hasBias {
			setAxon(i, expectedSplit, layer.Biases[i])
		}
	}
	return nil
}

// axonWeights collects the live Weight scalars from an axon bundle.
// Defined as a free helper so the extract path keeps a flat structure
// and the slice-of-cells iteration stays readable.
func axonWeights(axons axon.Bundle[float32]) []float32 {
	w := make([]float32, len(axons))
	for i, a := range axons {
		w[i] = a.Weight
	}
	return w
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
