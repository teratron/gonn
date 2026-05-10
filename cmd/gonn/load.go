package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/persistence"
	"github.com/teratron/gonn/pkg/utils"
)

// loadNetwork reads a config.json (and optionally a weights.json) from disk
// and returns a compiled, weight-installed *nn.NN[T] ready for inference or
// further training. When weightsPath is empty the returned network has
// randomly-initialised weights (suitable for training from scratch).
func loadNetwork[T utils.Float](cfgPath, weightsPath string) (*nn.NN[T], persistence.ConfigDoc[T], error) {
	doc, err := persistence.ReadConfig[T](cfgPath)
	if err != nil {
		return nil, doc, err
	}

	n, err := buildFromConfig[T](doc)
	if err != nil {
		return nil, doc, err
	}

	if weightsPath != "" {
		_, w, err := persistence.ReadWeights[T](cfgPath, weightsPath)
		if err != nil {
			return nil, doc, err
		}
		if err := installWeights(n, doc, w); err != nil {
			return nil, doc, err
		}
	}

	return n, doc, nil
}

// saveWeights extracts trained weights from n and writes them to path.
func saveWeights[T utils.Float](path string, cfg persistence.ConfigDoc[T], n *nn.NN[T]) error {
	w := extractWeights(n, cfg)
	return persistence.WriteWeights(path, cfg, w)
}

// buildFromConfig compiles a blank (randomly initialised) NN from a ConfigDoc.
func buildFromConfig[T utils.Float](doc persistence.ConfigDoc[T]) (*nn.NN[T], error) {
	b := nn.NewBuilder[T]().Input(doc.InputSize)

	for _, h := range doc.HiddenLayers {
		act, err := resolveActivation(h.Activation)
		if err != nil {
			return nil, err
		}
		b = b.Dense(h.Size, act, h.Bias)
	}

	outAct, err := resolveActivation(doc.Output.Activation)
	if err != nil {
		return nil, err
	}

	lossType, err := resolveLoss(doc.Training.Loss)
	if err != nil {
		return nil, err
	}

	wInit, err := resolveWeightInit(doc.Training.WeightInit)
	if err != nil {
		return nil, err
	}

	return b.
		Output(doc.Output.Size, outAct, doc.Output.Bias).
		WithLearningRate(doc.Training.LearningRate).
		WithLoss(lossType).
		WithMaxIterations(doc.Training.MaxIterations).
		WithLossLimit(doc.Training.LossLimit).
		WithWeightInit(wInit).
		Compile()
}

// installWeights copies weights from a WeightsDoc into a compiled network.
func installWeights[T utils.Float](n *nn.NN[T], doc persistence.ConfigDoc[T], w persistence.WeightsDoc[T]) error {
	want := len(doc.HiddenLayers) + 1
	if len(w.Layers) != want {
		return utils.Newf(utils.ErrIntegrity,
			"installWeights: expected %d layers (hidden chain + output), got %d", want, len(w.Layers))
	}

	for i, h := range doc.HiddenLayers {
		cells := n.Network.Hiddens[i].Cells()
		if err := installLayerG(fmt.Sprintf("hidden_%d", i), h.Bias, len(cells), w.Layers[i],
			func(ci, ai int, val T) { cells[ci].Axons[ai].Weight = val },
			func(ci int) int { return len(cells[ci].Axons) },
		); err != nil {
			return err
		}
	}

	outCells := n.Network.Output.Cells()
	return installLayerG("output", doc.Output.Bias, len(outCells), w.Layers[len(w.Layers)-1],
		func(ci, ai int, val T) { outCells[ci].Axons[ai].Weight = val },
		func(ci int) int { return len(outCells[ci].Axons) },
	)
}

// extractWeights packages the live axon weights from a compiled network into
// a WeightsDoc. The bias axon is the last axon in each cell's bundle when the
// layer declares bias == true.
func extractWeights[T utils.Float](n *nn.NN[T], doc persistence.ConfigDoc[T]) persistence.WeightsDoc[T] {
	layers := make([]persistence.LayerWeights[T], 0, len(doc.HiddenLayers)+1)

	for i, h := range doc.HiddenLayers {
		cells := n.Network.Hiddens[i].Cells()
		layers = append(layers, extractLayerG(
			fmt.Sprintf("hidden_%d", i), h.Bias, len(cells),
			func(ci int) []T {
				w := make([]T, len(cells[ci].Axons))
				for j, a := range cells[ci].Axons {
					w[j] = a.Weight
				}
				return w
			},
		))
	}

	outCells := n.Network.Output.Cells()
	layers = append(layers, extractLayerG(
		"output", doc.Output.Bias, len(outCells),
		func(ci int) []T {
			w := make([]T, len(outCells[ci].Axons))
			for j, a := range outCells[ci].Axons {
				w[j] = a.Weight
			}
			return w
		},
	))

	return persistence.WeightsDoc[T]{Layers: layers}
}

// installLayerG is the generic weight-install helper shared by hidden and
// output layers. The setAxon and axonCount callbacks close over the live
// cell slices so this function stays free of the cell type split.
func installLayerG[T utils.Float](
	name string,
	hasBias bool,
	numCells int,
	layer persistence.LayerWeights[T],
	setAxon func(cellIdx, axonIdx int, w T),
	axonCount func(cellIdx int) int,
) error {
	if len(layer.Weights) != numCells {
		return utils.Newf(utils.ErrIntegrity,
			"installLayerG %s: doc has %d cells, net has %d", name, len(layer.Weights), numCells)
	}
	if hasBias && len(layer.Biases) != numCells {
		return utils.Newf(utils.ErrIntegrity,
			"installLayerG %s: bias count mismatch — doc %d, net %d", name, len(layer.Biases), numCells)
	}
	for i := range numCells {
		row := layer.Weights[i]
		expectedSplit := axonCount(i)
		if hasBias {
			expectedSplit--
		}
		if len(row) != expectedSplit {
			return utils.Newf(utils.ErrIntegrity,
				"installLayerG %s[%d]: weight row width %d != expected %d", name, i, len(row), expectedSplit)
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

// extractLayerG packages one layer's weights into a LayerWeights[T].
func extractLayerG[T utils.Float](
	name string,
	hasBias bool,
	numCells int,
	axonsForCell func(int) []T,
) persistence.LayerWeights[T] {
	out := persistence.LayerWeights[T]{
		Name:    name,
		Weights: make([][]T, numCells),
	}
	if hasBias {
		out.Biases = make([]T, numCells)
	}
	for i := range numCells {
		ws := axonsForCell(i)
		split := len(ws)
		if hasBias {
			split--
		}
		row := make([]T, split)
		copy(row, ws[:split])
		out.Weights[i] = row
		if hasBias {
			out.Biases[i] = ws[split]
		}
	}
	return out
}

// resolveActivation maps the canonical String() name back to activation.Type.
// Mirrors the unexported persistence.resolveActivation.
func resolveActivation(name string) (activation.Type, error) {
	for t := activation.Type(0); t <= activation.TanH; t++ {
		if t.String() == name {
			return t, nil
		}
	}
	return 0, utils.NewActivationError(name)
}

// resolveLoss maps the canonical String() name back to loss.Type.
// Mirrors the unexported persistence.resolveLoss.
func resolveLoss(name string) (loss.Type, error) {
	for t := loss.Type(0); t <= loss.ARCTAN; t++ {
		if t.String() == name {
			return t, nil
		}
	}
	return 0, utils.Newf(utils.ErrUserConfig, "loss %q is not registered", name)
}

// resolveWeightInit converts the on-disk string tag to a WeightInitMethod.
// Empty string yields the library default (Xavier).
func resolveWeightInit(name string) (nn.WeightInitMethod, error) {
	switch nn.WeightInitMethod(name) {
	case nn.WeightInitXavier, nn.WeightInitHe, nn.WeightInitRandom:
		return nn.WeightInitMethod(name), nil
	case "":
		return nn.WeightInitXavier, nil
	default:
		return "", utils.Newf(utils.ErrUserConfig, "weight_init %q is not registered", name)
	}
}
