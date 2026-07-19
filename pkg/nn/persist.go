// Package nn — public Save / Load persistence API.
//
// Promotes the extract/install glue that previously lived in cmd/gonn and
// the persistence example into the facade (audit B5: "pkg/nn does not yet
// expose dump / load hooks"). The on-disk artefacts are the two documents
// owned by pkg/persistence: config.json (architecture + hyperparameters)
// and weights.json (trained values, hash-linked to the config).
//
// Conv-prefix networks are rejected: persistence schema 1.1.0 describes
// dense topologies only. Attempting to Save one errs loudly instead of
// silently dropping the prefix layers.
package nn

import (
	"fmt"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/persistence"
	"github.com/teratron/gonn/pkg/utils"
)

// Save writes the network's architecture and/or trained weights to disk.
// Either path may be empty to skip that artefact; passing both empty is an
// error. The weights document embeds a hash of the config document, so a
// later [Load] (or the gonn CLI) can detect config/weights drift.
//
// Networks constructed by [Load] reuse the exact config document read from
// disk, keeping the hash chain intact across train → save → query cycles
// even when the on-disk config predates this library version.
//
// AI-Meta:
//   - Purpose: Persist a compiled network's config and weights via pkg/persistence.
//   - Usage: err := n.Save("config.json", "weights.json").
//   - Concurrency: ReadSafe; takes the read lock while extracting weights.
//   - Errors: ErrUserConfig (not Operational, conv prefix, both paths empty), ErrIO (write failure).
//   - Related: [Load], [persistence.WriteConfig], [persistence.WriteWeights].
//   - Stability: Stable.
func (n *NN[T]) Save(configPath, weightsPath string) error {
	if n.stateField != stateOperational {
		return utils.Newf(utils.ErrUserConfig,
			"Save: network is %s, must be Operational", n.stateField.String())
	}
	if configPath == "" && weightsPath == "" {
		return utils.Newf(utils.ErrUserConfig,
			"Save: at least one of configPath / weightsPath must be non-empty")
	}
	if len(n.convPrefix) > 0 {
		return utils.Newf(utils.ErrUserConfig,
			"Save: conv/recurrent prefix layers are not covered by persistence schema %s (dense topologies only)",
			persistence.SchemaVersion)
	}

	doc := n.persistenceDoc()
	if configPath != "" {
		if err := persistence.WriteConfig(configPath, doc); err != nil {
			return err
		}
	}
	if weightsPath != "" {
		n.mu.RLock()
		w := extractWeightsDoc(n, doc)
		n.mu.RUnlock()
		if err := persistence.WriteWeights(weightsPath, doc, w); err != nil {
			return err
		}
	}
	return nil
}

// Load reads a config document (and optionally a weights document) from disk
// and returns a compiled *NN[T]. With an empty weightsPath the network keeps
// its randomly initialised weights — suitable for training from scratch.
//
// AI-Meta:
//   - Purpose: Reconstruct a ready-to-use network from persisted config and weights.
//   - Usage: n, err := nn.Load[float32]("config.json", "weights.json").
//   - Concurrency: SingleGoroutine.
//   - Errors: ErrIO (read failure), ErrUserConfig (schema/enum mismatch), ErrIntegrity (weight shape/hash drift).
//   - Related: [NN.Save], [persistence.ReadConfig], [persistence.ReadWeights].
//   - Stability: Stable.
func Load[T utils.Float](configPath, weightsPath string) (*NN[T], error) {
	doc, err := persistence.ReadConfig[T](configPath)
	if err != nil {
		return nil, err
	}
	n, err := buildFromConfigDoc(doc)
	if err != nil {
		return nil, err
	}
	// Cache the on-disk document so a later Save re-emits byte-identical
	// canonical config bytes — keeping the weights-doc hash chain intact.
	n.persistDoc = &doc

	if weightsPath != "" {
		_, w, err := persistence.ReadWeights[T](configPath, weightsPath)
		if err != nil {
			return nil, err
		}
		if err := installWeightsDoc(n, doc, w); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// persistenceDoc returns the ConfigDoc describing this network: the cached
// on-disk document when the network came from Load, otherwise a fresh
// projection of the compiled configuration.
func (n *NN[T]) persistenceDoc() persistence.ConfigDoc[T] {
	if n.persistDoc != nil {
		return *n.persistDoc
	}
	cfg := n.cfg
	hid := make([]persistence.HiddenLayerDoc, len(cfg.HiddenLayers))
	for i, h := range cfg.HiddenLayers {
		hid[i] = persistence.HiddenLayerDoc{
			Activation: h.Activation.String(),
			Size:       h.Size,
			Bias:       h.Bias,
		}
	}
	doc := persistence.ConfigDoc[T]{
		SchemaVersion: persistence.SchemaVersion,
		LibVersion:    libVersion,
		Training: persistence.TrainingDoc[T]{
			LearningRate:  cfg.LearningRate,
			LossLimit:     cfg.LossLimit,
			Loss:          cfg.LossType.String(),
			WeightInit:    string(cfg.WeightInit),
			MaxIterations: cfg.MaxIterations,
		},
		HiddenLayers: hid,
		Output: persistence.OutputDoc{
			Activation: cfg.OutputActivation.String(),
			Size:       cfg.OutputSize,
			Bias:       cfg.OutputBias,
		},
		InputSize: cfg.InputSize,
	}
	n.persistDoc = &doc
	return doc
}

// buildFromConfigDoc compiles a blank (randomly initialised) NN from a
// ConfigDoc read off disk.
func buildFromConfigDoc[T utils.Float](doc persistence.ConfigDoc[T]) (*NN[T], error) {
	b := NewBuilder[T]().Input(doc.InputSize)

	for _, h := range doc.HiddenLayers {
		act, err := resolveActivationName(h.Activation)
		if err != nil {
			return nil, err
		}
		b = b.Dense(h.Size, act, h.Bias)
	}

	outAct, err := resolveActivationName(doc.Output.Activation)
	if err != nil {
		return nil, err
	}
	lossType, err := resolveLossName(doc.Training.Loss)
	if err != nil {
		return nil, err
	}
	wInit, err := resolveWeightInitName(doc.Training.WeightInit)
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

// installWeightsDoc copies weights from a WeightsDoc into a compiled network.
func installWeightsDoc[T utils.Float](n *NN[T], doc persistence.ConfigDoc[T], w persistence.WeightsDoc[T]) error {
	want := len(doc.HiddenLayers) + 1
	if len(w.Layers) != want {
		return utils.Newf(utils.ErrIntegrity,
			"installWeightsDoc: expected %d layers (hidden chain + output), got %d", want, len(w.Layers))
	}

	for i, h := range doc.HiddenLayers {
		cells := n.Network.Hiddens[i].Cells()
		if err := installLayerDoc(fmt.Sprintf("hidden_%d", i), h.Bias, len(cells), w.Layers[i],
			func(ci, ai int, val T) { cells[ci].Axons[ai].SetW(val) },
			func(ci int) int { return len(cells[ci].Axons) },
		); err != nil {
			return err
		}
	}

	outCells := n.Network.Output.Cells()
	return installLayerDoc("output", doc.Output.Bias, len(outCells), w.Layers[len(w.Layers)-1],
		func(ci, ai int, val T) { outCells[ci].Axons[ai].SetW(val) },
		func(ci int) int { return len(outCells[ci].Axons) },
	)
}

// extractWeightsDoc packages the live axon weights from a compiled network
// into a WeightsDoc. The bias axon is the last axon in each cell's bundle
// when the layer declares bias == true.
func extractWeightsDoc[T utils.Float](n *NN[T], doc persistence.ConfigDoc[T]) persistence.WeightsDoc[T] {
	layers := make([]persistence.LayerWeights[T], 0, len(doc.HiddenLayers)+1)

	for i, h := range doc.HiddenLayers {
		cells := n.Network.Hiddens[i].Cells()
		layers = append(layers, extractLayerDoc(
			fmt.Sprintf("hidden_%d", i), h.Bias, len(cells),
			func(ci int) []T {
				w := make([]T, len(cells[ci].Axons))
				for j, a := range cells[ci].Axons {
					w[j] = a.W()
				}
				return w
			},
		))
	}

	outCells := n.Network.Output.Cells()
	layers = append(layers, extractLayerDoc(
		"output", doc.Output.Bias, len(outCells),
		func(ci int) []T {
			w := make([]T, len(outCells[ci].Axons))
			for j, a := range outCells[ci].Axons {
				w[j] = a.W()
			}
			return w
		},
	))

	return persistence.WeightsDoc[T]{Layers: layers}
}

// installLayerDoc is the generic weight-install helper shared by hidden and
// output layers. The setAxon and axonCount callbacks close over the live
// cell slices so this function stays free of the cell type split.
func installLayerDoc[T utils.Float](
	name string,
	hasBias bool,
	numCells int,
	layer persistence.LayerWeights[T],
	setAxon func(cellIdx, axonIdx int, w T),
	axonCount func(cellIdx int) int,
) error {
	if len(layer.Weights) != numCells {
		return utils.Newf(utils.ErrIntegrity,
			"installLayerDoc %s: doc has %d cells, net has %d", name, len(layer.Weights), numCells)
	}
	if hasBias && len(layer.Biases) != numCells {
		return utils.Newf(utils.ErrIntegrity,
			"installLayerDoc %s: bias count mismatch — doc %d, net %d", name, len(layer.Biases), numCells)
	}
	for i := range numCells {
		row := layer.Weights[i]
		expectedSplit := axonCount(i)
		if hasBias {
			expectedSplit--
		}
		if len(row) != expectedSplit {
			return utils.Newf(utils.ErrIntegrity,
				"installLayerDoc %s[%d]: weight row width %d != expected %d", name, i, len(row), expectedSplit)
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

// extractLayerDoc packages one layer's weights into a LayerWeights[T].
func extractLayerDoc[T utils.Float](
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

// resolveActivationName maps the canonical String() name back to
// activation.Type.
func resolveActivationName(name string) (activation.Type, error) {
	for t := activation.Type(0); t <= activation.TanH; t++ {
		if t.String() == name {
			return t, nil
		}
	}
	return 0, utils.NewActivationError(name)
}

// resolveLossName maps the canonical String() name back to loss.Type.
func resolveLossName(name string) (loss.Type, error) {
	for t := loss.Type(0); t <= loss.ARCTAN; t++ {
		if t.String() == name {
			return t, nil
		}
	}
	return 0, utils.Newf(utils.ErrUserConfig, "loss %q is not registered", name)
}

// resolveWeightInitName converts the on-disk string tag to a
// WeightInitMethod. Empty string yields the library default (Xavier).
func resolveWeightInitName(name string) (WeightInitMethod, error) {
	switch WeightInitMethod(name) {
	case WeightInitXavier, WeightInitHe, WeightInitRandom:
		return WeightInitMethod(name), nil
	case "":
		return WeightInitXavier, nil
	default:
		return "", utils.Newf(utils.ErrUserConfig, "weight_init %q is not registered", name)
	}
}
