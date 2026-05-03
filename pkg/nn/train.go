package nn

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Sample is a convenience alias for one (input, target) pair fed into
// Fit. Two parallel slices keep the type signature simple and avoid a
// dedicated struct that would need to carry the generic parameter.
type Sample[T utils.Float] struct {
	Input  []T
	Target []T
}

// Train runs one forward + backward + weight-update step for the supplied
// (input, target) pair using the network's configured learning rate and
// loss function. Returns the loss measured on this single sample.
//
// Single-step semantics — the multi-epoch loop with early stopping and
// min-loss snapshot lives in [Fit]. Train is the lower-level primitive
// for callers that own their own outer loop (the legacy perceptron
// example follows this style).
//
// Returns ErrUserConfig when the network is not Operational, or
// whatever Network.Train returns (typically ErrInputData on shape
// mismatch).
func (n *NN[T]) Train(input, target []T) (T, error) {
	if n.stateField != stateOperational {
		return 0, utils.Newf(utils.ErrUserConfig,
			"Train: network is %s, must be Operational", n.stateField.String())
	}
	return n.Network.Train(input, target)
}

// Fit runs the multi-epoch training loop driven by the cfg.MaxIterations
// and cfg.LossLimit early-stopping criteria from [l1-training-semantics].
// One epoch processes every sample in dataset once, in order; the per-
// epoch loss is the arithmetic mean of per-sample losses.
//
// Snapshot / rollback ([l2-training-loop] §snapshot mechanics): every
// time the running loss reaches a new minimum, Fit records a deep copy
// of the network weights. If the final epoch ends with a loss higher
// than the recorded minimum, Fit restores the snapshot before returning.
// This guards against late-epoch divergence wiping out a good model.
//
// Returns the number of epochs executed and the lowest mean-epoch loss
// observed. EpochCallback is invoked synchronously at the end of every
// epoch with (epoch, lossValue); a nil callback is skipped.
//
// Honours pause / stop signals between epochs (see [control.go]).
func (n *NN[T]) Fit(dataset []Sample[T]) (uint, T, error) {
	if n.stateField != stateOperational {
		return 0, 0, utils.Newf(utils.ErrUserConfig,
			"Fit: network is %s, must be Operational", n.stateField.String())
	}
	if len(dataset) == 0 {
		return 0, 0, utils.Newf(utils.ErrInputData,
			"Fit: dataset must contain at least one sample")
	}

	n.transitionToRunning()
	defer n.transitionToIdle()

	minLoss := T(0)
	minLossSet := false
	var snapshot []T
	var lastLoss T
	var completedEpochs uint

	for epoch := uint(1); epoch <= n.cfg.MaxIterations; epoch++ {
		// Safe-point check before each epoch — honours Pause / Stop
		// transitions issued from another goroutine.
		if stopped, err := n.awaitSafePoint(); err != nil {
			return completedEpochs, lastLoss, err
		} else if stopped {
			break
		}

		var total T
		for batchIdx, sample := range dataset {
			loss, err := n.Network.Train(sample.Input, sample.Target)
			if err != nil {
				return completedEpochs, lastLoss, err
			}
			total += loss
			if cb := n.cfg.BatchCallback; cb != nil {
				cb(uint(batchIdx), loss)
			}
		}
		mean := total / T(len(dataset))
		lastLoss = mean
		completedEpochs = epoch
		if cb := n.cfg.EpochCallback; cb != nil {
			cb(epoch, mean)
		}

		if !minLossSet || mean < minLoss {
			minLoss = mean
			minLossSet = true
			snapshot = n.snapshotWeights(snapshot)
			if mean < n.cfg.LossLimit {
				return completedEpochs, mean, nil
			}
		}
	}

	// Late-epoch divergence guard: if the final epoch's loss is worse
	// than the best observed, roll back to the snapshot. Cheap pointer
	// assignment per axon — no allocation churn.
	if minLossSet && lastLoss > minLoss && snapshot != nil {
		n.restoreWeights(snapshot)
		lastLoss = minLoss
	}
	return completedEpochs, lastLoss, nil
}

// snapshotWeights writes every axon weight into dst (re-using the slice
// when its capacity already covers the network) and returns the result.
// Reusing the slice avoids per-epoch GC pressure on long training runs.
func (n *NN[T]) snapshotWeights(dst []T) []T {
	count := n.weightCount()
	if cap(dst) < count {
		dst = make([]T, count)
	} else {
		dst = dst[:count]
	}
	idx := 0
	for _, hb := range n.Network.Hiddens {
		for _, h := range hb.Cells() {
			for _, a := range h.Axons {
				dst[idx] = a.Weight
				idx++
			}
		}
	}
	for _, o := range n.Network.Output.Cells() {
		for _, a := range o.Axons {
			dst[idx] = a.Weight
			idx++
		}
	}
	return dst
}

// restoreWeights writes a previously captured snapshot back into the
// network. The snapshot must have been produced by snapshotWeights on
// the same topology — Fit owns this invariant.
func (n *NN[T]) restoreWeights(src []T) {
	idx := 0
	for _, hb := range n.Network.Hiddens {
		for _, h := range hb.Cells() {
			for i := range h.Axons {
				h.Axons[i].Weight = src[idx]
				idx++
			}
		}
	}
	for _, o := range n.Network.Output.Cells() {
		for i := range o.Axons {
			o.Axons[i].Weight = src[idx]
			idx++
		}
	}
}

// weightCount totals the number of trainable scalars across hidden and
// output axons. Used to size the snapshot buffer.
func (n *NN[T]) weightCount() int {
	count := 0
	for _, hb := range n.Network.Hiddens {
		for _, h := range hb.Cells() {
			count += len(h.Axons)
		}
	}
	for _, o := range n.Network.Output.Cells() {
		count += len(o.Axons)
	}
	return count
}
