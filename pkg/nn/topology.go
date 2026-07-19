package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

// onTopologyChanged runs after a successful runtime topology mutation. The
// trainable weight count has changed, so the optimizer's moment/velocity state
// and the cached flat weight/gradient buffers are stale. Reset the optimizer
// and drop the buffers so the next training step rebuilds them at the new size,
// preventing the index-out-of-range panic the old code hit (audit D3/D4).
// Callers hold n.mu.
func (n *NN[T]) onTopologyChanged() {
	if n.opt != nil {
		n.opt.Reset()
	}
	n.weightBuf = nil
	n.gradBuf = nil
	n.convGradBuf = nil
}

// requireIdleOrPaused returns ErrControl when training is actively running.
// Topology mutations are only safe when the training loop is not executing.
func (n *NN[T]) requireIdleOrPaused() error {
	c := n.control.Load()
	if c != controlIdle && c != controlPaused {
		return utils.Newf(utils.ErrControl,
			"topology mutation requires Idle or Paused training state; current state code: %d", c)
	}
	return nil
}

// AddNeuron adds count neurons to the hidden layer at layerIdx.
// Enforces DYN-2: fails when training is actively running (controlRunning).
// Delegates mutation to the embedded Network[T].
//
// AI-Meta:
//   - Purpose: DYN-2-gated wrapper around Network.AddNeuron for safe runtime topology growth.
//   - Errors: ErrControl (running), ErrImmutableMode, ErrInvalidPosition, ErrEmptyLayer.
//   - Related: [network.Network.AddNeuron], [RemoveNeuron], [TopologyVersion].
//   - Stability: Stable.
func (n *NN[T]) AddNeuron(layerIdx, count uint) error {
	if err := n.requireIdleOrPaused(); err != nil {
		return err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if err := n.Network.AddNeuron(layerIdx, count); err != nil {
		return err
	}
	n.onTopologyChanged()
	return nil
}

// RemoveNeuron removes count neurons from the end of the hidden layer at layerIdx.
// Enforces DYN-2: fails when training is actively running.
//
// AI-Meta:
//   - Purpose: DYN-2-gated wrapper around Network.RemoveNeuron.
//   - Errors: ErrControl (running), ErrImmutableMode, ErrInvalidPosition, ErrEmptyLayer.
//   - Related: [network.Network.RemoveNeuron], [AddNeuron].
//   - Stability: Stable.
func (n *NN[T]) RemoveNeuron(layerIdx, count uint) error {
	if err := n.requireIdleOrPaused(); err != nil {
		return err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if err := n.Network.RemoveNeuron(layerIdx, count); err != nil {
		return err
	}
	n.onTopologyChanged()
	return nil
}

// AddHiddenLayer inserts a new hidden layer at position in the chain.
// Enforces DYN-2: fails when training is actively running.
//
// AI-Meta:
//   - Purpose: DYN-2-gated wrapper around Network.AddHiddenLayer.
//   - Errors: ErrControl (running), ErrImmutableMode, ErrInvalidPosition, ErrEmptyLayer.
//   - Related: [network.Network.AddHiddenLayer], [RemoveHiddenLayer].
//   - Stability: Stable.
func (n *NN[T]) AddHiddenLayer(position, size uint, act activation.Type, bias bool) error {
	if err := n.requireIdleOrPaused(); err != nil {
		return err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if err := n.Network.AddHiddenLayer(position, size, act, bias); err != nil {
		return err
	}
	n.onTopologyChanged()
	return nil
}

// RemoveHiddenLayer removes the hidden layer at position.
// Enforces DYN-2: fails when training is actively running.
//
// AI-Meta:
//   - Purpose: DYN-2-gated wrapper around Network.RemoveHiddenLayer.
//   - Errors: ErrControl (running), ErrImmutableMode, ErrInvalidPosition, ErrMinimumTopology.
//   - Related: [network.Network.RemoveHiddenLayer], [AddHiddenLayer].
//   - Stability: Stable.
func (n *NN[T]) RemoveHiddenLayer(position uint) error {
	if err := n.requireIdleOrPaused(); err != nil {
		return err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if err := n.Network.RemoveHiddenLayer(position); err != nil {
		return err
	}
	n.onTopologyChanged()
	return nil
}
