package network

import (
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/utils"
)

// denseLayer is the structure-of-arrays view of one fully connected layer
// (a hidden layer, or the output layer). The pointer graph still defines the
// TOPOLOGY — which source cell feeds which slot — but the weights themselves
// live in one contiguous run per layer, carved out of the network-wide
// [Network.weights] array. That is what lets a compute backend consume a layer
// as a plain matrix (see [compute.DenseKernels]) instead of chasing pointers.
//
// Layout is row-major [out][in]: slot (o, j) is at store.W[o*in+j]. This is
// exactly the canonical flat-weight order produced by AppendFlatWeights
// (layer → cell → axon), so FlatWeights/ApplyFlatWeights degenerate to a copy
// over the contiguous array.
//
// The trailing input column is the bias when hasBias is set: input[in-1] is
// pinned to 1, which folds the bias term into the same matrix-vector product
// instead of special-casing it in every kernel.
type denseLayer[T utils.Float] struct {
	store   *axon.Store[T]
	in      int // fan-in INCLUDING the bias column when hasBias
	out     int
	hasBias bool

	// Scratch buffers, allocated once per topology and reused every sample.
	input  []T // source activations, length in (input[in-1] == 1 when hasBias)
	preact []T // pre-activation sums, length out (aliases preactHiddens[i] / preactOutput)
	act    []T // post-activation → post-norm → post-mask values, length out
	miss   []T // −∂L/∂act, length out
	delta  []T // −∂L/∂preact = σ′(preact) ⊙ miss, length out
	dInput []T // −∂L/∂input propagated to the previous layer, length in
}

// buildStore (re)allocates contiguous weight storage for every dense layer and
// binds each axon to its slot, preserving the axon's current weight. Called by
// Build and after every topology mutation — the axons keep their identity, only
// the backing storage is rebuilt, so learned weights survive.
//
// Returns an error when a layer's axon bundles are ragged (cells within one
// layer disagreeing on fan-in), which would make the matrix view unsound.
func (n *Network[T]) buildStore() error {
	layerCount := len(n.Hiddens) + 1
	dims := make([][2]int, 0, layerCount) // (out, in) per layer

	for li, hb := range n.Hiddens {
		out := hb.Len()
		if out == 0 {
			return utils.Newf(utils.ErrUserConfig, "buildStore: Hiddens[%d] has no cells", li)
		}
		in := len(hb.cells[0].Axons)
		for ci, h := range hb.cells {
			if len(h.Axons) != in {
				return utils.Newf(utils.ErrUserConfig,
					"buildStore: ragged bundle in Hiddens[%d]: cell %d has %d axons, cell 0 has %d",
					li, ci, len(h.Axons), in)
			}
		}
		dims = append(dims, [2]int{out, in})
	}
	outCount := n.Output.Len()
	if outCount == 0 {
		return utils.Newf(utils.ErrUserConfig, "buildStore: output layer has no cells")
	}
	outIn := len(n.Output.cells[0].Axons)
	for ci, o := range n.Output.cells {
		if len(o.Axons) != outIn {
			return utils.Newf(utils.ErrUserConfig,
				"buildStore: ragged bundle in Output: cell %d has %d axons, cell 0 has %d",
				ci, len(o.Axons), outIn)
		}
	}
	dims = append(dims, [2]int{outCount, outIn})

	total := 0
	for _, d := range dims {
		total += d[0] * d[1]
	}
	// A fresh array plus fresh Store headers: Bind reads each axon's CURRENT
	// weight (from the OLD store when rebuilding) before repointing it, so this
	// is a safe hand-off rather than a reset.
	n.weights = make([]T, total)
	n.dense = make([]denseLayer[T], len(dims))

	off := 0
	for li, d := range dims {
		out, in := d[0], d[1]
		dl := &n.dense[li]
		dl.out, dl.in = out, in
		dl.store = &axon.Store[T]{W: n.weights[off : off+out*in : off+out*in]}
		off += out * in

		if li < len(n.Hiddens) {
			dl.hasBias = n.hiddenBiases[li] != nil
			dl.preact = n.preactHiddens[li]
			for ci, h := range n.Hiddens[li].cells {
				for ai := range h.Axons {
					h.Axons[ai].Bind(dl.store, ci*in+ai)
				}
			}
		} else {
			dl.hasBias = n.outputBias != nil
			dl.preact = n.preactOutput
			for ci, o := range n.Output.cells {
				for ai := range o.Axons {
					o.Axons[ai].Bind(dl.store, ci*in+ai)
				}
			}
		}

		dl.input = make([]T, in)
		if dl.hasBias {
			dl.input[in-1] = 1 // pinned; the bias column never changes
		}
		dl.act = make([]T, out)
		dl.miss = make([]T, out)
		dl.delta = make([]T, out)
		dl.dInput = make([]T, in)
	}
	return nil
}

// storeReady reports whether the SoA store matches the current topology. Guards
// the fast paths so a Network that was hand-assembled in a test (SetLayers +
// Build skipped, or cells mutated directly) still behaves.
func (n *Network[T]) storeReady() bool {
	if len(n.dense) != len(n.Hiddens)+1 {
		return false
	}
	for li, hb := range n.Hiddens {
		if n.dense[li].out != hb.Len() {
			return false
		}
	}
	return n.dense[len(n.dense)-1].out == n.Output.Len()
}

// sourceCount returns the number of REAL source cells feeding layer li — the
// fan-in minus the bias column. Used when propagating the input gradient back:
// the bias column has no upstream cell to receive it.
func (dl *denseLayer[T]) sourceCount() int {
	if dl.hasBias {
		return dl.in - 1
	}
	return dl.in
}

// loadInput copies src into the layer's input buffer, leaving the pinned bias
// column untouched.
func (dl *denseLayer[T]) loadInput(src []T) {
	copy(dl.input[:dl.sourceCount()], src)
}
