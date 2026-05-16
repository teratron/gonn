package network

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// TopologyMode controls whether runtime topology mutations are permitted.
//
// AI-Meta:
//   - Purpose: Enum for opting a Network into dynamic topology at construction time.
//   - Usage: pass network.Dynamic to WithTopologyMode in the NN builder/options API.
//   - Related: [Network.AddNeuron], [Network.AddHiddenLayer].
//   - Stability: Stable.
type TopologyMode uint8

const (
	// Immutable is the default — all mutation methods return ErrImmutableMode.
	Immutable TopologyMode = iota

	// Dynamic enables AddNeuron / RemoveNeuron / AddHiddenLayer / RemoveHiddenLayer.
	Dynamic
)

// topologyTx is a shallow snapshot of topology-owned slices taken before a
// mutation begins. Rollback restores all slices to their pre-mutation values;
// Commit increments the topology version counter.
type topologyTx[T utils.Float] struct {
	n             *Network[T]
	hiddens       []bundle[T, *cell.Hidden[T]]
	hiddenBiases  []*cell.Bias[T]
	hiddenActs    []activation.Type
	preactHiddens [][]T
}

// beginTx snapshots the current topology state and returns a transaction handle.
func (n *Network[T]) beginTx() topologyTx[T] {
	hCopy := make([]bundle[T, *cell.Hidden[T]], len(n.Hiddens))
	copy(hCopy, n.Hiddens)
	bCopy := make([]*cell.Bias[T], len(n.hiddenBiases))
	copy(bCopy, n.hiddenBiases)
	aCopy := make([]activation.Type, len(n.hiddenActs))
	copy(aCopy, n.hiddenActs)
	pCopy := make([][]T, len(n.preactHiddens))
	copy(pCopy, n.preactHiddens)
	return topologyTx[T]{
		n:             n,
		hiddens:       hCopy,
		hiddenBiases:  bCopy,
		hiddenActs:    aCopy,
		preactHiddens: pCopy,
	}
}

// Commit increments the topology version after a successful mutation.
func (tx *topologyTx[T]) Commit() {
	tx.n.topologyVersion.Add(1)
}

// Rollback restores all topology slices to their pre-mutation values.
func (tx *topologyTx[T]) Rollback() {
	tx.n.Hiddens = tx.hiddens
	tx.n.hiddenBiases = tx.hiddenBiases
	tx.n.hiddenActs = tx.hiddenActs
	tx.n.preactHiddens = tx.preactHiddens
}

// SetTopologyMode sets the topology mode at compile time. Called by nn.compile
// after Build completes so the network begins life in the configured mode.
//
// AI-Meta:
//   - Purpose: Configure TopologyMode at compile time; called by nn.compile.
//   - Related: [TopologyMode], [requireDynamic].
//   - Stability: Stable.
func (n *Network[T]) SetTopologyMode(mode TopologyMode) {
	n.topologyMode = mode
}

// requireDynamic returns ErrImmutableMode when the network is in Immutable mode (DYN-1).
func (n *Network[T]) requireDynamic() error {
	if n.topologyMode == Immutable {
		return utils.Newf(utils.ErrImmutableMode,
			"network topology is immutable; opt in with network.Dynamic via WithTopologyMode")
	}
	return nil
}

// TopologyVersion returns the monotonic counter incremented by every
// successful topology mutation commit. Zero on freshly-built networks.
//
// AI-Meta:
//   - Purpose: Read-only topology change counter for snapshot invalidation.
//   - Concurrency: Safe; uses atomic load.
//   - Related: [AddNeuron], [AddHiddenLayer].
//   - Stability: Stable.
func (n *Network[T]) TopologyVersion() uint64 {
	return n.topologyVersion.Load()
}

// AddNeuron adds count new cells to the hidden layer at layerIdx and
// re-wires its incoming and outgoing axons. Returns an error when:
//   - topologyMode == Immutable (ErrImmutableMode)
//   - layerIdx is out of range (ErrInvalidPosition)
//   - count == 0 (ErrEmptyLayer)
//
// On any error the topology is rolled back to its pre-call state.
//
// AI-Meta:
//   - Purpose: Grow a hidden layer by count neurons and rebalance adjacent axons.
//   - Errors: ErrImmutableMode, ErrInvalidPosition, ErrEmptyLayer.
//   - Concurrency: NotSafe; caller must ensure no concurrent Train/Query.
//   - Related: [RemoveNeuron], [AddHiddenLayer], [TopologyVersion].
//   - Stability: Stable.
func (n *Network[T]) AddNeuron(layerIdx, count uint) error {
	if err := n.requireDynamic(); err != nil {
		return err
	}
	if int(layerIdx) >= len(n.Hiddens) {
		return utils.Newf(utils.ErrInvalidPosition,
			"AddNeuron: layerIdx %d out of range [0, %d)", layerIdx, len(n.Hiddens))
	}
	if count == 0 {
		return utils.Newf(utils.ErrEmptyLayer, "AddNeuron: count must be > 0")
	}
	tx := n.beginTx()

	oldCells := n.Hiddens[layerIdx].cells
	newCells := make([]*cell.Hidden[T], len(oldCells)+int(count))
	copy(newCells, oldCells)
	for i := range count {
		newCells[len(oldCells)+int(i)] = cell.NewHidden[T](uint(len(oldCells) + int(i)))
	}
	n.Hiddens[layerIdx].cells = newCells
	n.preactHiddens[layerIdx] = make([]T, len(newCells))

	n.rebalance(int(layerIdx))
	if int(layerIdx)+1 < len(n.Hiddens) {
		n.rebalance(int(layerIdx) + 1)
	} else {
		n.rebalanceOutput()
	}
	tx.Commit()
	return nil
}

// RemoveNeuron removes count cells from the end of the hidden layer at
// layerIdx and re-wires adjacent axons. Returns an error when:
//   - topologyMode == Immutable (ErrImmutableMode)
//   - layerIdx is out of range (ErrInvalidPosition)
//   - count == 0 (ErrEmptyLayer)
//   - the resulting layer size would be zero (ErrEmptyLayer)
//
// On any error the topology is rolled back.
//
// AI-Meta:
//   - Purpose: Shrink a hidden layer by count neurons and rebalance adjacent axons.
//   - Errors: ErrImmutableMode, ErrInvalidPosition, ErrEmptyLayer.
//   - Concurrency: NotSafe; caller must ensure no concurrent Train/Query.
//   - Related: [AddNeuron], [RemoveHiddenLayer].
//   - Stability: Stable.
func (n *Network[T]) RemoveNeuron(layerIdx, count uint) error {
	if err := n.requireDynamic(); err != nil {
		return err
	}
	if int(layerIdx) >= len(n.Hiddens) {
		return utils.Newf(utils.ErrInvalidPosition,
			"RemoveNeuron: layerIdx %d out of range [0, %d)", layerIdx, len(n.Hiddens))
	}
	if count == 0 {
		return utils.Newf(utils.ErrEmptyLayer, "RemoveNeuron: count must be > 0")
	}
	cur := len(n.Hiddens[layerIdx].cells)
	if int(count) >= cur {
		return utils.Newf(utils.ErrEmptyLayer,
			"RemoveNeuron: removing %d from layer %d (size %d) would leave it empty",
			count, layerIdx, cur)
	}
	tx := n.beginTx()

	n.Hiddens[layerIdx].cells = n.Hiddens[layerIdx].cells[:cur-int(count)]
	n.preactHiddens[layerIdx] = make([]T, len(n.Hiddens[layerIdx].cells))

	n.rebalance(int(layerIdx))
	if int(layerIdx)+1 < len(n.Hiddens) {
		n.rebalance(int(layerIdx) + 1)
	} else {
		n.rebalanceOutput()
	}
	tx.Commit()
	return nil
}

// AddHiddenLayer inserts a new Dense hidden layer of the given size and
// activation at position in the Hiddens chain. The predecessor and successor
// axons are re-wired. Returns an error when:
//   - topologyMode == Immutable (ErrImmutableMode)
//   - position > len(Hiddens) (ErrInvalidPosition)
//   - size == 0 (ErrEmptyLayer)
//
// AI-Meta:
//   - Purpose: Insert a new hidden layer at position and rebalance surrounding axons.
//   - Errors: ErrImmutableMode, ErrInvalidPosition, ErrEmptyLayer.
//   - Concurrency: NotSafe; caller must ensure no concurrent Train/Query.
//   - Related: [RemoveHiddenLayer], [AddNeuron].
//   - Stability: Stable.
func (n *Network[T]) AddHiddenLayer(position, size uint, act activation.Type, bias bool) error {
	if err := n.requireDynamic(); err != nil {
		return err
	}
	if int(position) > len(n.Hiddens) {
		return utils.Newf(utils.ErrInvalidPosition,
			"AddHiddenLayer: position %d out of range [0, %d]", position, len(n.Hiddens))
	}
	if size == 0 {
		return utils.Newf(utils.ErrEmptyLayer, "AddHiddenLayer: size must be > 0")
	}
	tx := n.beginTx()

	// Build new cells for the inserted layer.
	newCells := make([]*cell.Hidden[T], int(size))
	for i := range newCells {
		newCells[i] = cell.NewHidden[T](uint(i))
	}
	newBundle := newBundle[T, *cell.Hidden[T]]()
	newBundle.cells = newCells

	var newBias *cell.Bias[T]
	if bias {
		b := cell.NewBias[T]()
		newBias = b
	}

	// Insert into positional slices at position.
	pos := int(position)
	n.Hiddens = append(n.Hiddens[:pos], append([]bundle[T, *cell.Hidden[T]]{newBundle}, n.Hiddens[pos:]...)...)
	n.hiddenBiases = append(n.hiddenBiases[:pos], append([]*cell.Bias[T]{newBias}, n.hiddenBiases[pos:]...)...)
	n.hiddenActs = append(n.hiddenActs[:pos], append([]activation.Type{act}, n.hiddenActs[pos:]...)...)
	n.preactHiddens = append(n.preactHiddens[:pos], append([][]T{make([]T, int(size))}, n.preactHiddens[pos:]...)...)

	// Rebalance the new layer and its successor.
	n.rebalance(pos)
	if pos+1 < len(n.Hiddens) {
		n.rebalance(pos + 1)
	} else {
		n.rebalanceOutput()
	}
	tx.Commit()
	return nil
}

// RemoveHiddenLayer removes the hidden layer at position, bridges the
// predecessor and successor layers, and re-wires axons. Returns an error when:
//   - topologyMode == Immutable (ErrImmutableMode)
//   - position is out of range (ErrInvalidPosition)
//   - the network has only one hidden layer (ErrMinimumTopology)
//
// AI-Meta:
//   - Purpose: Remove a hidden layer and bridge surrounding layers.
//   - Errors: ErrImmutableMode, ErrInvalidPosition, ErrMinimumTopology.
//   - Concurrency: NotSafe; caller must ensure no concurrent Train/Query.
//   - Related: [AddHiddenLayer], [RemoveNeuron].
//   - Stability: Stable.
func (n *Network[T]) RemoveHiddenLayer(position uint) error {
	if err := n.requireDynamic(); err != nil {
		return err
	}
	if len(n.Hiddens) <= 1 {
		return utils.Newf(utils.ErrMinimumTopology,
			"RemoveHiddenLayer: network must retain at least one hidden layer")
	}
	if int(position) >= len(n.Hiddens) {
		return utils.Newf(utils.ErrInvalidPosition,
			"RemoveHiddenLayer: position %d out of range [0, %d)", position, len(n.Hiddens))
	}
	tx := n.beginTx()

	pos := int(position)
	n.Hiddens = append(n.Hiddens[:pos], n.Hiddens[pos+1:]...)
	n.hiddenBiases = append(n.hiddenBiases[:pos], n.hiddenBiases[pos+1:]...)
	n.hiddenActs = append(n.hiddenActs[:pos], n.hiddenActs[pos+1:]...)
	n.preactHiddens = append(n.preactHiddens[:pos], n.preactHiddens[pos+1:]...)

	// Rebalance the layer that is now at position (successor of removed).
	if pos < len(n.Hiddens) {
		n.rebalance(pos)
	} else {
		// Removed last hidden layer; rebalance Output from new last hidden.
		n.rebalanceOutput()
	}
	tx.Commit()
	return nil
}

// rebalance re-wires all incoming axons for Hiddens[layerIdx]. Called after
// any structural change to a layer or its predecessor.
func (n *Network[T]) rebalance(layerIdx int) {
	hb := &n.Hiddens[layerIdx]
	bias := n.hiddenBiases[layerIdx]
	fanOut := hb.Len()

	for _, h := range hb.cells {
		h.Axons = h.Axons[:0]
		if layerIdx == 0 {
			fanIn := n.Input.Len()
			for _, src := range n.Input.cells {
				h.Axons = append(h.Axons, n.newAxon(src, h, fanIn, fanOut))
			}
		} else {
			prev := &n.Hiddens[layerIdx-1]
			fanIn := prev.Len()
			for _, src := range prev.cells {
				h.Axons = append(h.Axons, n.newAxon(src, h, fanIn, fanOut))
			}
		}
		if bias != nil {
			h.Axons = append(h.Axons, n.newAxon(bias, h, 1, fanOut))
		}
	}
}

// rebalanceOutput re-wires all incoming axons for the Output bundle from
// the last hidden layer. Called after mutations that affect the last hidden.
func (n *Network[T]) rebalanceOutput() {
	lastHidden := &n.Hiddens[len(n.Hiddens)-1]
	fanIn := lastHidden.Len()
	fanOut := n.Output.Len()
	for _, o := range n.Output.cells {
		o.Axons = o.Axons[:0]
		for _, src := range lastHidden.cells {
			o.Axons = append(o.Axons, n.newAxon(src, o, fanIn, fanOut))
		}
		if n.outputBias != nil {
			o.Axons = append(o.Axons, n.newAxon(n.outputBias, o, 1, fanOut))
		}
	}
}
