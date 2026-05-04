// Package network — the internal computational graph.
//
// Network[T] is the value-typed engine that owns three cell groups
// (Input, a chain of Hiddens, Output), wires axons between them, and
// runs the forward / backward / weight-update pipeline. Public entry
// points (Builder, NN[T]) are out of scope for Phase 1 — the facade
// layer is restored in Phase 2 and the multi-hidden chain in Phase 5.
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
//
// Per [l2-multihidden-impl] §5.1 the Hidden field generalises to a slice
// of bundles in compile-time order. Single-hidden topologies (v0.5)
// continue to work unchanged: callers wrap their one Dense layer in a
// one-element slice when calling SetLayers.
type Network[T utils.Float] struct {
	LearningRate T `json:"learningRate" xml:"learningRate"`

	Input   bundle[T, *cell.Input[T]]    `json:"input" xml:"input"`
	Hiddens []bundle[T, *cell.Hidden[T]] `json:"hiddens" xml:"hiddens"`
	Output  bundle[T, *cell.Output[T]]   `json:"output" xml:"output"`

	// hiddenBiases / hiddenActs are positional with Hiddens[i]: a layer
	// with Bias == false contributes a nil entry so the lookup stays
	// indexable without auxiliary maps.
	hiddenBiases []*cell.Bias[T]
	outputBias   *cell.Bias[T]
	hiddenActs   []activation.Type
	outputAct    activation.Type
	lossMode     loss.Type

	// preactHiddens[i] / preactOutput store the pre-activation linear
	// sums so backprop can feed them to activation.Derivative. The
	// dispatcher expects pre-activation input (it re-applies the
	// activation inside to compute σ' = σ(x)·(1-σ(x)) for sigmoid and
	// similar). Reusing the post-activation value would yield
	// σ(σ(x))·(1-σ(σ(x))) — a vanishing gradient that prevents
	// convergence. preactHiddens[i] is sized to Hiddens[i].Len() once at
	// SetLayers time and never reslicing afterwards.
	preactHiddens [][]T
	preactOutput  []T
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
		Hiddens:      nil,
		Output:       newBundle[T, *cell.Output[T]](),
	}
}

// SetLayers installs cells from constructed layer values into the network
// bundles. The layer types own the cells; the network mirrors their
// slices so propagation methods can reach them. Bias cells declared by
// layers are stored separately (positionally aligned with Hiddens[i])
// and used as axon sources by Build.
//
// hiddens is the multi-hidden chain in left-to-right order. v0.5 callers
// pass a one-element slice; v0.6 supports any positive length.
//
// Returns an error wrapping ErrUserConfig when any required layer is
// nil, when the hiddens slice is empty, or when any layer reports zero
// size.
func (n *Network[T]) SetLayers(in *layer.Input[T], hiddens []*layer.Dense[T], out *layer.Output[T]) error {
	if in == nil || out == nil {
		return utils.Newf(utils.ErrUserConfig,
			"SetLayers: input and output layers must be non-nil (in=%v out=%v)",
			in != nil, out != nil,
		)
	}
	if len(hiddens) == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"SetLayers: hiddens slice must contain at least one layer (got 0)")
	}
	for i, h := range hiddens {
		if h == nil {
			return utils.Newf(utils.ErrUserConfig,
				"SetLayers: hiddens[%d] is nil — every chain entry must be a constructed Dense layer", i)
		}
		if h.Size == 0 {
			return utils.Newf(utils.ErrUserConfig,
				"SetLayers: hiddens[%d] has zero size — every Dense layer must declare Size > 0", i)
		}
	}
	if in.Size == 0 || out.Size == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"SetLayers: input and output layers must have positive size (in=%d out=%d)",
			in.Size, out.Size,
		)
	}

	n.Input.Replace(in.Cells())
	n.Output.Replace(out.Cells())

	// Resize positional metadata to match the new chain. Allocating
	// fresh slices (rather than mutating in place) keeps SetLayers
	// idempotent under retries — a previous call's longer chain does
	// not leak into the new one.
	n.Hiddens = make([]bundle[T, *cell.Hidden[T]], len(hiddens))
	n.hiddenBiases = make([]*cell.Bias[T], len(hiddens))
	n.hiddenActs = make([]activation.Type, len(hiddens))
	n.preactHiddens = make([][]T, len(hiddens))
	for i, h := range hiddens {
		// Hidden[T] is a generic alias of Dense[T]; the slice element
		// types are identical so the slice rebind is type-safe.
		n.Hiddens[i] = newBundle[T, *cell.Hidden[T]]()
		n.Hiddens[i].Replace(h.Cells())
		n.hiddenBiases[i] = h.BiasCell()
		n.hiddenActs[i] = h.Activation
		n.preactHiddens[i] = make([]T, h.Size)
	}

	n.outputBias = out.BiasCell()
	n.outputAct = out.Activation
	n.lossMode = out.Loss
	n.preactOutput = make([]T, out.Size)
	return nil
}

// LossMode reports the loss-function symbol the Output layer was built
// with. Surfaced for [Network.CalculateLossDefault] callers that want
// the configured mode without re-walking the layer chain.
func (n *Network[T]) LossMode() loss.Type {
	return n.lossMode
}

// Build wires axons across the Input → Hiddens → Output chain.
//
// Per [l2-multihidden-impl] §5.3:
//   - Hiddens[0] cells receive one axon per Input cell (and one from
//     hiddenBiases[0] if present).
//   - Hiddens[i] cells (i ≥ 1) receive one axon per Hiddens[i-1] cell
//     (and one from hiddenBiases[i] if present).
//   - Output cells receive one axon per Hiddens[len-1] cell (and one
//     from outputBias if present).
//
// Subsequent calls overwrite the existing axon bundles — Build is
// idempotent and safe under dynamic-topology adjustments.
func (n *Network[T]) Build() error {
	if n.Input.Len() == 0 || len(n.Hiddens) == 0 || n.Output.Len() == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"Build: empty bundle (in=%d hiddenChain=%d out=%d) — call SetLayers first",
			n.Input.Len(), len(n.Hiddens), n.Output.Len(),
		)
	}
	for i := range n.Hiddens {
		if n.Hiddens[i].Len() == 0 {
			return utils.Newf(utils.ErrUserConfig,
				"Build: Hiddens[%d] has zero cells — call SetLayers with non-empty layers", i)
		}
	}
	for i, hb := range n.Hiddens {
		bias := n.hiddenBiases[i]
		for _, h := range hb.cells {
			h.Axons = h.Axons[:0]
			if i == 0 {
				for _, src := range n.Input.cells {
					h.Axons = append(h.Axons, axon.New[T](src, h))
				}
			} else {
				for _, src := range n.Hiddens[i-1].cells {
					h.Axons = append(h.Axons, axon.New[T](src, h))
				}
			}
			if bias != nil {
				h.Axons = append(h.Axons, axon.New[T](bias, h))
			}
		}
	}
	lastHidden := n.Hiddens[len(n.Hiddens)-1].cells
	for _, o := range n.Output.cells {
		o.Axons = o.Axons[:0]
		for _, src := range lastHidden {
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
