// Package network — the internal computational graph.
//
// Network[T] is the value-typed engine that owns three cell bundles
// (Input, Hidden, Output), wires axons between them, and runs the
// forward / backward / weight-update pipeline. Public entry points
// (Builder, NN[T]) are out of scope for Phase 1 — the facade layer
// is restored in Phase 2.
package network

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// defaultLearningRate matches [l2-network-graph] §2 — 0.3 baseline.
const defaultLearningRate = 0.3

// Network is the typed graph. Embedded by [pkg/nn].NN in Phase 2.
type Network[T utils.Float] struct {
	LearningRate T `json:"learningRate" xml:"learningRate"`

	Input  bundle[T, *cell.Input[T]]  `json:"input" xml:"input"`
	Hidden bundle[T, *cell.Hidden[T]] `json:"hidden" xml:"hidden"`
	Output bundle[T, *cell.Output[T]] `json:"output" xml:"output"`

	hiddenBias *cell.Bias[T]
	outputBias *cell.Bias[T]
	hiddenAct  activation.Type
	outputAct  activation.Type
	lossMode   loss.Type
}

// New returns a freshly constructed Network with empty bundles and the
// default learning rate. Returned by value so the [pkg/nn].NN facade can
// embed it directly per [l2-nn-facade] §5.1; callers that mutate state
// must do so through pointer-receiver methods (which Go addresses
// automatically when the value is addressable).
func New[T utils.Float]() Network[T] {
	return Network[T]{
		LearningRate: T(defaultLearningRate),
		Input:        newBundle[T, *cell.Input[T]](),
		Hidden:       newBundle[T, *cell.Hidden[T]](),
		Output:       newBundle[T, *cell.Output[T]](),
	}
}

// SetLayers installs cells from constructed layer values into the network
// bundles. The layer types own the cells; the network mirrors their
// slices so propagation methods can reach them. Bias cells declared by
// layers are stored separately and used as axon sources by Build.
//
// Returns an error wrapping ErrUserConfig when any required layer is nil.
func (n *Network[T]) SetLayers(in *layer.Input[T], hidden *layer.Dense[T], out *layer.Output[T]) error {
	if in == nil || hidden == nil || out == nil {
		return utils.Newf(utils.ErrUserConfig,
			"SetLayers: all three layers must be non-nil (in=%v hidden=%v out=%v)",
			in != nil, hidden != nil, out != nil,
		)
	}
	if in.Size == 0 || hidden.Size == 0 || out.Size == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"SetLayers: every layer must have positive size (in=%d hidden=%d out=%d)",
			in.Size, hidden.Size, out.Size,
		)
	}
	n.Input.Replace(in.Cells())
	// Hidden[T] is a generic alias of Dense[T]; the slice element types
	// are identical so the slice rebind is type-safe.
	n.Hidden.Replace(hidden.Cells())
	n.Output.Replace(out.Cells())
	n.hiddenBias = hidden.BiasCell()
	n.outputBias = out.BiasCell()
	n.hiddenAct = hidden.Activation
	n.outputAct = out.Activation
	n.lossMode = out.Loss
	return nil
}

// LossMode reports the loss-function symbol the Output layer was built
// with. Surfaced for [Network.CalculateLossDefault] callers that want
// the configured mode without re-walking the layer chain.
func (n *Network[T]) LossMode() loss.Type {
	return n.lossMode
}

// Build wires axons between layers. Each Hidden cell receives one axon
// per Input cell (and one from hiddenBias if present); each Output cell
// receives one axon per Hidden cell (and one from outputBias if present).
// Subsequent calls overwrite the existing axon bundles — Build is
// idempotent and safe under dynamic-topology adjustments.
func (n *Network[T]) Build() error {
	if n.Input.Len() == 0 || n.Hidden.Len() == 0 || n.Output.Len() == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"Build: empty bundle (in=%d hidden=%d out=%d) — call SetLayers first",
			n.Input.Len(), n.Hidden.Len(), n.Output.Len(),
		)
	}
	for _, h := range n.Hidden.cells {
		h.Axons = h.Axons[:0]
		for _, src := range n.Input.cells {
			h.Axons = append(h.Axons, axon.New[T](src, h))
		}
		if n.hiddenBias != nil {
			h.Axons = append(h.Axons, axon.New[T](n.hiddenBias, h))
		}
	}
	for _, o := range n.Output.cells {
		o.Axons = o.Axons[:0]
		for _, src := range n.Hidden.cells {
			o.Axons = append(o.Axons, axon.New[T](src, o))
		}
		if n.outputBias != nil {
			o.Axons = append(o.Axons, axon.New[T](n.outputBias, o))
		}
	}
	return nil
}

// SetInputs writes one sample into the Input bundle. The slice length
// must match the bundle size; mismatched lengths return ErrInputData.
func (n *Network[T]) SetInputs(data []T) error {
	if len(data) != n.Input.Len() {
		return utils.Newf(utils.ErrInputData,
			"SetInputs: expected %d values, got %d", n.Input.Len(), len(data),
		)
	}
	for idx, v := range data {
		n.Input.cells[idx].SetValue(v)
	}
	return nil
}

// SetTargets writes the label vector into the Output cells. Same shape
// constraint as SetInputs.
func (n *Network[T]) SetTargets(data []T) error {
	if len(data) != n.Output.Len() {
		return utils.Newf(utils.ErrInputData,
			"SetTargets: expected %d values, got %d", n.Output.Len(), len(data),
		)
	}
	for idx, v := range data {
		// cell.Output stores its target via pointer; assign through the
		// pointer to keep the cell's existing reference valid.
		*n.Output.cells[idx].GetTarget() = v
	}
	return nil
}
