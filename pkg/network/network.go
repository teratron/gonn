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
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// defaultLearningRate matches [l2-network-graph] §2 — 0.3 baseline.
const defaultLearningRate = 0.3

// WeightSampler is the function signature used by SetWeightSampler.
// fanIn is the previous-layer size; fanOut is the current-layer size.
// Returning T(0) is valid (e.g., zero-init for bias connections).
//
// AI-Meta:
//   - Purpose: Callback type for weight initialisation strategies supplied via SetWeightSampler.
//   - Usage: n.SetWeightSampler(func(fanIn, fanOut int) float32 { return utils.XavierUniform[float32](rng, fanIn, fanOut) }).
//   - Related: [Network.SetWeightSampler], [Network.Build].
//   - Stability: Stable.
type WeightSampler[T utils.Float] func(fanIn, fanOut int) T

// Network is the typed computational graph. Embedded by nn.NN so the
// public facade delegates forward/backward passes without an extra heap
// allocation. Owns three bundle groups (Input, Hiddens chain, Output),
// their bias cells, activation tags, and pre-activation scratch buffers
// for backprop.
//
// AI-Meta:
//   - Purpose: Internal engine owning the full neural graph; forward, backward, and weight-update steps.
//   - Lifecycle: Zero → populated via SetLayers + Build → operational via Train/CalculateValues.
//   - Concurrency: NotSafe; Train mutates cells, weights, and pre-activation buffers in place.
//   - Related: [New], [SetLayers], [Build], [Train], [CalculateValues].
type Network[T utils.Float] struct {
	LearningRate    T `json:"learningRate" xml:"learningRate"`
	outputBias      *cell.Bias[T]
	initWeight      WeightSampler[T]
	hiddenBiases    []*cell.Bias[T]
	Input           bundle[T, *cell.Input[T]] `json:"input" xml:"input"`
	hiddenActs      []activation.Type
	preactHiddens   [][]T
	preactOutput    []T
	Output          bundle[T, *cell.Output[T]]   `json:"output" xml:"output"`
	Hiddens         []bundle[T, *cell.Hidden[T]] `json:"hiddens" xml:"hiddens"`
	topologyVersion uint64
	outputAct       activation.Type
	lossMode        loss.Type
	topologyMode    TopologyMode
}

// New returns a freshly constructed Network with empty bundles and the
// default learning rate (0.3). Returned by value so nn.NN can embed it
// directly; pointer-receiver methods are reachable once the value is
// addressable.
//
// AI-Meta:
//   - Purpose: Allocate an empty Network ready for SetLayers + Build.
//   - Usage: n := network.New[float32](); n.SetLayers(...); n.Build(); n.Train(...).
//   - Related: [Network], [SetLayers], [Build].
func New[T utils.Float]() Network[T] {
	return Network[T]{
		LearningRate: T(defaultLearningRate),
		Input:        newBundle[T, *cell.Input[T]](),
		Hiddens:      nil,
		Output:       newBundle[T, *cell.Output[T]](),
	}
}

// SetWeightSampler configures the weight-initialization function applied by
// subsequent Build calls. nil reverts to the axon.New default (U[-0.5, 0.5]).
// Call this after SetLayers and before Build.
//
// AI-Meta:
//   - Purpose: Override the default weight initializer; enables Xavier/He/Uniform sampling.
//   - Usage: n.SetWeightSampler(func(fanIn, fanOut int) float32 { return utils.XavierUniform[float32](rng, fanIn, fanOut) }).
//   - Related: [WeightSampler], [Build].
//   - Stability: Stable.
func (n *Network[T]) SetWeightSampler(fn WeightSampler[T]) {
	n.initWeight = fn
}

// SetLayers installs the constructed layer values into the network bundles.
// Layer types own the cells; the network mirrors their slices so propagation
// methods can reach them. Bias cells are stored positionally and used as axon
// sources by Build.
//
// hiddens is the left-to-right hidden chain; must contain at least one layer.
//
// AI-Meta:
//   - Purpose: Wire layer cells into the network graph before calling Build.
//   - Errors: ErrUserConfig (nil layer, empty hiddens, zero-size layer).
//   - Concurrency: NotSafe; must complete before Build.
//   - Related: [Network], [Build], [layer.Input], [layer.Dense], [layer.Output].
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

// LossMode reports the loss-function symbol the Output layer was built with.
//
// AI-Meta:
//   - Purpose: Expose the configured loss type so callers can pass it to CalculateLoss explicitly.
//   - Concurrency: Safe; read-only after SetLayers.
//   - Related: [CalculateLoss], [CalculateLossDefault].
func (n *Network[T]) LossMode() loss.Type {
	return n.lossMode
}

// Build wires axons across the Input → Hiddens → Output chain. Each cell
// in Hiddens[i] receives one incoming axon per cell in the previous layer
// (Input for i=0, Hiddens[i-1] otherwise) plus one from its bias cell if
// present; Output cells connect from the last hidden layer plus bias.
// Subsequent calls overwrite existing axon bundles — idempotent.
//
// AI-Meta:
//   - Purpose: Wire all axons after SetLayers; required before any forward pass.
//   - Errors: ErrUserConfig (empty bundles, not called after SetLayers).
//   - Concurrency: NotSafe; must complete before Train or CalculateValues.
//   - Related: [SetLayers], [Network], [axon.New].
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
		fanOut := hb.Len()
		for _, h := range hb.cells {
			h.Axons = h.Axons[:0]
			if i == 0 {
				fanIn := n.Input.Len()
				for _, src := range n.Input.cells {
					h.Axons = append(h.Axons, n.newAxon(src, h, fanIn, fanOut))
				}
			} else {
				fanIn := n.Hiddens[i-1].Len()
				for _, src := range n.Hiddens[i-1].cells {
					h.Axons = append(h.Axons, n.newAxon(src, h, fanIn, fanOut))
				}
			}
			if bias != nil {
				h.Axons = append(h.Axons, n.newAxon(bias, h, 1, fanOut))
			}
		}
	}
	lastHidden := n.Hiddens[len(n.Hiddens)-1].cells
	fanIn := n.Hiddens[len(n.Hiddens)-1].Len()
	fanOut := n.Output.Len()
	for _, o := range n.Output.cells {
		o.Axons = o.Axons[:0]
		for _, src := range lastHidden {
			o.Axons = append(o.Axons, n.newAxon(src, o, fanIn, fanOut))
		}
		if n.outputBias != nil {
			o.Axons = append(o.Axons, n.newAxon(n.outputBias, o, 1, fanOut))
		}
	}
	return nil
}

// newAxon creates an axon with a weight drawn from initWeight when set,
// or falls back to axon.New (U[-0.5, 0.5]) for backward compatibility.
func (n *Network[T]) newAxon(
	src neuron.Nucleus[T],
	dst neuron.Neuron[T],
	fanIn, fanOut int,
) *axon.Axon[T] {
	if n.initWeight != nil {
		return axon.NewWithWeight(n.initWeight(fanIn, fanOut), src, dst)
	}
	return axon.New[T](src, dst)
}

// SetInputs writes one sample into the Input bundle. Slice length must
// match the bundle size.
//
// AI-Meta:
//   - Purpose: Load one feature vector into input cells before CalculateValues.
//   - Errors: ErrInputData (length mismatch).
//   - Concurrency: NotSafe; must complete before CalculateValues.
//   - Related: [SetTargets], [Train].
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

// SetTargets writes the label vector into Output cell target pointers.
// Slice length must match the output bundle size.
//
// AI-Meta:
//   - Purpose: Load ground-truth labels for the current sample before CalculateValues.
//   - Errors: ErrInputData (length mismatch).
//   - Concurrency: NotSafe; must complete before CalculateValues.
//   - Related: [SetInputs], [Train].
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

// FlatWeights collects every learnable weight (hidden + output axons) into a
// newly allocated flat slice in the canonical order: Hiddens[0] → Hiddens[n-1]
// → Output. Used by the optimizer integration in pkg/nn.
//
// AI-Meta:
//   - Purpose: Export a flat weight copy for optimizer Step calls.
//   - Related: [ApplyFlatWeights], [FlatGradients].
//   - Stability: Stable.
func (n *Network[T]) FlatWeights() []T {
	out := make([]T, 0, n.weightCount())
	for _, hb := range n.Hiddens {
		for _, h := range hb.cells {
			for _, a := range h.Axons {
				out = append(out, a.Weight)
			}
		}
	}
	for _, o := range n.Output.cells {
		for _, a := range o.Axons {
			out = append(out, a.Weight)
		}
	}
	return out
}

// AppendFlatWeights is like FlatWeights but appends into dst (reusing its
// backing array when capacity is sufficient). Returns the extended slice.
//
// AI-Meta:
//   - Purpose: Zero-alloc variant of FlatWeights for hot-path training loops.
//   - Related: [FlatWeights], [ApplyFlatWeights].
//   - Stability: Stable.
func (n *Network[T]) AppendFlatWeights(dst []T) []T {
	dst = dst[:0]
	for _, hb := range n.Hiddens {
		for _, h := range hb.cells {
			for _, a := range h.Axons {
				dst = append(dst, a.Weight)
			}
		}
	}
	for _, o := range n.Output.cells {
		for _, a := range o.Axons {
			dst = append(dst, a.Weight)
		}
	}
	return dst
}

// ApplyFlatWeights writes a flat weight slice back in the same canonical order
// produced by FlatWeights / AppendFlatWeights. The slice must have the same
// length; a length mismatch silently truncates (caller invariant).
//
// AI-Meta:
//   - Purpose: Write optimizer-updated weights back into the network graph.
//   - Related: [FlatWeights], [AppendFlatWeights].
//   - Stability: Stable.
func (n *Network[T]) ApplyFlatWeights(weights []T) {
	idx := 0
	for _, hb := range n.Hiddens {
		for _, h := range hb.cells {
			for i := range h.Axons {
				if idx >= len(weights) {
					return
				}
				h.Axons[i].Weight = weights[idx]
				idx++
			}
		}
	}
	for _, o := range n.Output.cells {
		for i := range o.Axons {
			if idx >= len(weights) {
				return
			}
			o.Axons[i].Weight = weights[idx]
			idx++
		}
	}
}

// weightCount totals all learnable axon weights across hidden and output layers.
func (n *Network[T]) weightCount() int {
	count := 0
	for _, hb := range n.Hiddens {
		for _, h := range hb.cells {
			count += len(h.Axons)
		}
	}
	for _, o := range n.Output.cells {
		count += len(o.Axons)
	}
	return count
}

// HiddenActivations collects all hidden-layer post-activation values into a flat
// slice: Hiddens[0] cells → Hiddens[n-1] cells. Used by the regularizer mask path.
//
// AI-Meta:
//   - Purpose: Export hidden activations for ApplyMask calls in the training loop.
//   - Related: [SetHiddenActivations].
//   - Stability: Stable.
func (n *Network[T]) HiddenActivations() []T {
	total := 0
	for _, hb := range n.Hiddens {
		total += hb.Len()
	}
	out := make([]T, 0, total)
	for _, hb := range n.Hiddens {
		for _, h := range hb.cells {
			out = append(out, *h.GetValue())
		}
	}
	return out
}

// SetHiddenActivations writes a flat activation slice back in the same order
// as HiddenActivations. Called after ApplyMask to persist dropout masks.
//
// AI-Meta:
//   - Purpose: Write regularizer-masked hidden activations back into cell values.
//   - Related: [HiddenActivations].
//   - Stability: Stable.
func (n *Network[T]) SetHiddenActivations(acts []T) {
	idx := 0
	for _, hb := range n.Hiddens {
		for _, h := range hb.cells {
			if idx >= len(acts) {
				return
			}
			*h.GetValue() = acts[idx]
			idx++
		}
	}
}
