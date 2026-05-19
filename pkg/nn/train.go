package nn

import (
	"slices"

	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/regularizer"
	"github.com/teratron/gonn/pkg/utils"
)

// libVersion is the GoNN library version embedded in structured log events.
const libVersion = "0.8.0"

// Sample is one (input, target) pair for use with Fit. Two parallel slices
// keep the type signature simple while preserving the generic parameter T.
//
// AI-Meta:
//   - Purpose: Typed container for one training sample passed to Fit.
//   - Usage: Fit([]nn.Sample[float32]{{Input: x, Target: y}, ...}).
//   - Concurrency: NotSafe.
//   - Related: [Fit].
//   - Stability: Stable.
type Sample[T utils.Float] struct {
	Input  []T
	Target []T
}

// Train runs one forward + backward + weight-update step for the supplied
// (input, target) pair using the configured optimizer. Returns the loss
// measured on this single sample.
//
// The primitive for callers that own their own outer loop. For the managed
// multi-epoch loop with early stopping and best-weight rollback, use Fit.
//
// AI-Meta:
//   - Purpose: Single-step training primitive; caller drives the epoch loop.
//   - Usage: for _, s := range samples { loss, err := n.Train(s.Input, s.Target) }.
//   - Concurrency: SingleGoroutine; must not run concurrently with other Train/Fit calls.
//   - Errors: ErrUserConfig (not Operational), ErrInputData (shape mismatch).
//   - Related: [Fit], [Query].
//   - Stability: Stable.
func (n *NN[T]) Train(input, target []T) (T, error) {
	if n.stateField != stateOperational {
		return 0, utils.Newf(utils.ErrUserConfig,
			"Train: network is %s, must be Operational", n.stateField.String())
	}
	return n.trainStep(input, target)
}

// trainStep executes one forward + backward + optimizer step.
// Shared by Train and the inner loop of Fit. When a conv prefix is
// configured the input is first transformed through the conv stack; the
// chain's final output replaces the raw input as the Network's Input.
// Backward gradient flow runs symmetrically: Network.AppendInputGradient
// produces ∂L/∂(conv output), which is piped through the conv layers in
// reverse to accumulate conv-weight gradients (CONV-4).
func (n *NN[T]) trainStep(input, target []T) (T, error) {
	netInput, err := n.runConvForward(input)
	if err != nil {
		return 0, err
	}
	if err := n.Network.SetInputs(netInput); err != nil {
		return 0, err
	}
	if err := n.Network.SetTargets(target); err != nil {
		return 0, err
	}
	n.Network.CalculateValues()

	// Apply dropout / mask after forward pass (training=true).
	if n.reg != nil {
		acts := n.Network.HiddenActivations()
		acts = n.reg.ApplyMask(acts, true)
		n.Network.SetHiddenActivations(acts)
	}

	lossVal := n.Network.CalculateLossDefault()

	// Add regularization penalty to the reported loss.
	if n.reg != nil {
		n.weightBuf = n.Network.AppendFlatWeights(n.weightBuf)
		lossVal += regularizer.Penalty(n.reg, n.weightBuf)
	}

	n.Network.CalculateMisses()

	// Drive the conv backward pass BEFORE the optimizer overwrites the
	// neuron axon weights — the input-gradient formula reads those weights
	// to project the hidden-layer miss back onto the Input cells.
	if len(n.convPrefix) > 0 {
		n.convGradBuf = n.Network.AppendInputGradient(n.convGradBuf)
		n.applyConvBackward(n.convGradBuf)
	}

	// Collect weights and gradients, delegate update to the optimizer.
	n.weightBuf = n.Network.AppendFlatWeights(n.weightBuf)
	n.gradBuf = n.Network.AppendFlatGradients(n.gradBuf)
	if n.cfg.GradClipNorm > 0 {
		optimizer.ClipByGlobalNorm([][]T{n.gradBuf}, n.cfg.GradClipNorm)
	}
	if err := n.opt.Step(n.weightBuf, n.gradBuf); err != nil {
		return lossVal, err
	}
	n.Network.ApplyFlatWeights(n.weightBuf)

	return lossVal, nil
}

// runConvForward pushes the raw input vector through the conv prefix and
// returns the chain's final output. Returns the input unchanged when no
// conv prefix is configured (zero allocation, zero overhead).
//
// AI-Meta:
//   - Purpose: Run the conv prefix as a preprocessing stage before the Dense head.
//   - Concurrency: NotSafe; mutates per-layer scratch buffers.
//   - Related: [trainStep], [Query], [applyConvBackward].
func (n *NN[T]) runConvForward(input []T) ([]T, error) {
	if len(n.convPrefix) == 0 {
		return input, nil
	}
	if n.rawInputSize != 0 && uint(len(input)) != n.rawInputSize {
		return nil, utils.Newf(utils.ErrInputData,
			"conv prefix: input length %d does not match declared raw size %d",
			len(input), n.rawInputSize)
	}
	cur := input
	for _, cl := range n.convPrefix {
		cur = cl.Forward(cur)
	}
	n.convBuf = cur
	return cur, nil
}

// applyConvBackward walks the conv chain in reverse, propagating the
// upstream gradient and updating Conv1D kernel weights via an inline SGD
// step against the network's learning rate. Pool and Flatten layers carry
// no parameters — their Backward implementations just reshape / route the
// gradient through. Wiring the conv weights through optimizer.Optimizer
// is deferred to a future minor (v0.11) so this v0.10 release keeps the
// integration small and observable.
//
// AI-Meta:
//   - Purpose: Backward-pass conv stack with inline SGD weight update on Conv1D layers.
//   - Concurrency: NotSafe; reads gradient slices and mutates conv weights.
//   - Related: [trainStep], [conv.Conv1D.Backward], [conv.Conv1D.GradSlots].
func (n *NN[T]) applyConvBackward(gradOut []T) {
	upstream := gradOut
	for _, v := range slices.Backward(n.convPrefix) {
		next := v.Backward(upstream)
		if c1d, ok := v.(*conv.Conv1D[T]); ok {
			gW, gB := c1d.GradSlots()
			applyConvSGD(c1d.Weights, gW, n.LearningRate)
			if c1d.UseBias && len(gB) > 0 {
				applyConvSGD(c1d.Biases, gB, n.LearningRate)
			}
		}
		upstream = next
	}
}

// applyConvSGD performs w -= lr · g elementwise. Length parity is the
// caller's invariant — Conv1D.GradSlots guarantees identical lengths.
func applyConvSGD[T utils.Float](w, g []T, lr T) {
	if len(g) == 0 {
		return
	}
	for i := range w {
		w[i] -= lr * g[i]
	}
}

// Fit runs the managed multi-epoch training loop. Each epoch processes every
// Sample once in order; the epoch loss is the arithmetic mean of per-sample
// losses. Early stops when mean loss drops below LossLimit or MaxIterations
// is reached, or when a Stop signal is received.
//
// Best-weight rollback: whenever a new minimum is reached Fit snapshots the
// weights. If the final epoch's loss is higher than the minimum, the snapshot
// is restored before returning.
//
// AI-Meta:
//   - Purpose: Run the full training loop with early stopping, callbacks, and best-weight rollback.
//   - Usage: epochs, loss, err := n.Fit(samples).
//   - Concurrency: SingleGoroutine; Pause/Resume/Stop may be called from another goroutine.
//   - Errors: ErrUserConfig (not Operational), ErrInputData (sample shape mismatch).
//   - Related: [Train], [Pause], [Resume], [Stop], [Sample].
//   - Stability: Stable.
func (n *NN[T]) Fit(dataset []Sample[T]) (uint, T, error) {
	if n.stateField != stateOperational {
		return 0, 0, utils.Newf(utils.ErrUserConfig,
			"Fit: network is %s, must be Operational", n.stateField.String())
	}
	if len(dataset) == 0 {
		return 0, 0, utils.Newf(utils.ErrInputData,
			"Fit: dataset must contain at least one sample")
	}

	log := utils.NewGoLogger(n.cfg.Logger, libVersion, "")
	log.Info("training started", "max_iterations", n.cfg.MaxIterations)

	n.transitionToRunning()

	var completedEpochs uint
	var lastLoss T
	// stopReason and lastLoss are captured by pointer in fireOnTrainEnd so the
	// deferred call reads the final values at Fit return time (CB-8).
	var stopReason StopReason
	stopReasonSet := false
	defer func() {
		n.transitionToIdle()
		log.Info("training stopped")
		if n.callbacks != nil {
			sr := stopReason
			if !stopReasonSet {
				sr = StopMaxIterations
			}
			fireOnTrainEnd(n.callbacks, &sr, &completedEpochs, &lastLoss)
		}
	}()

	minLoss := T(0)
	minLossSet := false
	var snapshot []T

	for epoch := uint(1); epoch <= n.cfg.MaxIterations; epoch++ {
		// Safe-point check before each epoch — honours Pause / Stop
		// transitions issued from another goroutine.
		if stopped, err := n.awaitSafePoint(); err != nil {
			stopReason = StopLoopError
			stopReasonSet = true
			return completedEpochs, lastLoss, err
		} else if stopped {
			stopReason = StopExternalStop
			stopReasonSet = true
			break
		}

		var total T
		for batchIdx, sample := range dataset {
			loss, err := n.trainStep(sample.Input, sample.Target)
			if err != nil {
				stopReason = StopLoopError
				stopReasonSet = true
				return completedEpochs, lastLoss, err
			}
			total += loss
			if cb := n.cfg.BatchCallback; cb != nil {
				cb(uint(batchIdx), loss)
			}
			// Advance the LR scheduler at step granularity (e.g. warm-up).
			if n.sched != nil && n.sched.Granularity() == optimizer.PerStep {
				n.sched.Step()
			}
		}
		mean := total / T(len(dataset))
		lastLoss = mean
		completedEpochs = epoch
		log.Debug("epoch completed", "epoch", epoch, "loss", float64(mean))
		if cb := n.cfg.EpochCallback; cb != nil {
			cb(epoch, mean)
		}

		// Advance the LR scheduler at epoch granularity (LRS-1).
		// When the scheduler implements MetricScheduler, supply the epoch
		// loss so plateau-based strategies can detect stalls (T-9A04).
		if n.sched != nil && n.sched.Granularity() == optimizer.PerEpoch {
			if ms, ok := n.sched.(optimizer.MetricScheduler[T]); ok {
				ms.StepWithMetric(mean)
			} else {
				n.sched.Step()
			}
		}

		if !minLossSet || mean < minLoss {
			minLoss = mean
			minLossSet = true
			snapshot = n.snapshotWeights(snapshot)

			// Dispatch OnImprovementFound before OnIterationEnd (CB-9).
			if n.callbacks != nil {
				snap := snapshotFromWeights(snapshot, epoch)
				ctx := callbackContextFrom(int(epoch), mean, minLoss, int(epoch), snap)
				if err := fireEvent(n.callbacks.OnImprovementFound, ctx); err != nil {
					stopReason = StopCallback
					stopReasonSet = true
					n.restoreWeights(snapshot)
					lastLoss = minLoss
					return completedEpochs, lastLoss, nil
				}
			}

			if mean < n.cfg.LossLimit {
				stopReason = StopLossLimit
				stopReasonSet = true
				return completedEpochs, mean, nil
			}
		}

		// Meta-learner hook: advisory per-epoch hyperparameter update.
		// Executes after opt.Step (inside trainStep) and before OnIterationEnd.
		// Errors are logged at Warn level and do NOT abort training (META-advisory).
		if n.cfg.MetaLearner != nil {
			if mlErr := n.cfg.MetaLearner.step(mean, int(epoch), int(n.cfg.MaxIterations)); mlErr != nil {
				log.Warn("meta-learner step failed", "error", mlErr)
			}
		}

		// Dispatch OnIterationEnd after the weight update block (CB-9).
		if n.callbacks != nil {
			snap := snapshotFromWeights(snapshot, epoch)
			ctx := callbackContextFrom(int(epoch), mean, minLoss, int(epoch), snap)
			if err := fireEvent(n.callbacks.OnIterationEnd, ctx); err != nil {
				stopReason = StopCallback
				stopReasonSet = true
				if minLossSet && snapshot != nil {
					n.restoreWeights(snapshot)
					lastLoss = minLoss
				}
				return completedEpochs, lastLoss, nil
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
