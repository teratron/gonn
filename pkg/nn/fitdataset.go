// Package nn — streaming-dataset training loop.
//
// FitDataset bridges pkg/dataset's pull-source contract into the managed
// training loop (audit B10: the dataset package existed but no NN method
// consumed it). One epoch = drain the source to io.EOF, then Reset. Sources
// that cannot rewind (ErrUnsupported) train for exactly one epoch.
package nn

import (
	"context"
	"errors"
	"io"

	"github.com/teratron/gonn/pkg/dataset"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/utils"
)

// FitDataset runs the managed multi-epoch training loop over a streaming
// Dataset. Semantics mirror [Fit]: epoch-mean loss, LossLimit early stop,
// best-weight rollback, callbacks, scheduler advancement, and checkpointing
// all behave identically. Differences:
//
//   - ctx cancellation stops training at the next batch boundary and
//     returns ctx's error; callbacks observe StopContextCancel.
//   - An epoch ends when Next returns io.EOF; Reset then rewinds the
//     source. When Reset reports dataset.ErrUnsupported (true stream),
//     training completes after that single epoch.
//   - BatchCallback receives the batch index and the batch-mean loss
//     (Fit, which has no batching, passes per-sample values).
//
// AI-Meta:
//   - Purpose: Managed training loop over a pull-source Dataset with context cancellation.
//   - Usage: epochs, loss, err := n.FitDataset(ctx, ds).
//   - Concurrency: SingleGoroutine; Pause/Resume/Stop may be called from another goroutine.
//   - Errors: ErrUserConfig (not Operational, nil dataset), ErrInputData (empty epoch), ctx.Err() on cancellation.
//   - Related: [Fit], [dataset.Dataset], [dataset.Prefetch], [WithCheckpoint].
//   - Stability: Stable.
func (n *NN[T]) FitDataset(ctx context.Context, ds dataset.Dataset[T]) (uint, T, error) {
	if n.stateField != stateOperational {
		return 0, 0, utils.Newf(utils.ErrUserConfig,
			"FitDataset: network is %s, must be Operational", n.stateField.String())
	}
	if ds == nil {
		return 0, 0, utils.Newf(utils.ErrUserConfig, "FitDataset: dataset is nil")
	}

	n.mu.Lock()
	defer n.mu.Unlock()

	log := utils.NewGoLogger(n.cfg.Logger, libVersion, "")
	log.Info("dataset training started", "max_iterations", n.cfg.MaxIterations)

	n.transitionToRunning()

	var completedEpochs uint
	var lastLoss T
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
	var bestEpoch uint
	var snapshot []T

	for epoch := uint(1); epoch <= n.cfg.MaxIterations; epoch++ {
		if err := ctx.Err(); err != nil {
			stopReason = StopContextCancel
			stopReasonSet = true
			return completedEpochs, lastLoss, err
		}
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
		count := 0
		batchIdx := uint(0)
		for {
			batch, err := ds.Next(ctx)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				if ctx.Err() != nil {
					stopReason = StopContextCancel
				} else {
					stopReason = StopLoopError
				}
				stopReasonSet = true
				return completedEpochs, lastLoss, err
			}
			var batchTotal T
			for i := range batch.Inputs {
				lossVal, stepErr := n.trainStep(batch.Inputs[i], batch.Targets[i])
				if stepErr != nil {
					stopReason = StopLoopError
					stopReasonSet = true
					return completedEpochs, lastLoss, stepErr
				}
				total += lossVal
				batchTotal += lossVal
				count++
				if n.sched != nil && n.sched.Granularity() == optimizer.PerStep {
					n.sched.Step()
				}
			}
			if cb := n.cfg.BatchCallback; cb != nil && batch.Len() > 0 {
				cb(batchIdx, batchTotal/T(batch.Len()))
			}
			batchIdx++
		}
		if count == 0 {
			stopReason = StopLoopError
			stopReasonSet = true
			return completedEpochs, lastLoss, utils.Newf(utils.ErrInputData,
				"FitDataset: dataset produced no samples in epoch %d", epoch)
		}

		mean := total / T(count)
		lastLoss = mean
		completedEpochs = epoch
		log.Debug("epoch completed", "epoch", epoch, "loss", float64(mean), "samples", count)
		n.publishVisSnapshot(float64(mean), uint64(epoch), uint64(epoch)*uint64(count))
		if cb := n.cfg.EpochCallback; cb != nil {
			cb(epoch, mean)
		}

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
			bestEpoch = epoch
			snapshot = n.snapshotWeights(snapshot)

			if n.callbacks != nil {
				snap := snapshotFromWeights(snapshot, epoch)
				cbCtx := callbackContextFrom(int(epoch), mean, minLoss, int(epoch), snap)
				if err := fireEvent(n.callbacks.OnImprovementFound, cbCtx); err != nil {
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

		n.maybeWriteCheckpoint(epoch, mean, minLoss, bestEpoch)

		if n.callbacks != nil {
			snap := snapshotFromWeights(snapshot, epoch)
			cbCtx := callbackContextFrom(int(epoch), mean, minLoss, int(epoch), snap)
			if err := fireEvent(n.callbacks.OnIterationEnd, cbCtx); err != nil {
				stopReason = StopCallback
				stopReasonSet = true
				if minLossSet && snapshot != nil {
					n.restoreWeights(snapshot)
					lastLoss = minLoss
				}
				return completedEpochs, lastLoss, nil
			}
		}

		if epoch < n.cfg.MaxIterations {
			if err := ds.Reset(ctx); err != nil {
				if errors.Is(err, dataset.ErrUnsupported) {
					// True stream — one pass is all the data there is.
					break
				}
				if ctx.Err() != nil {
					stopReason = StopContextCancel
				} else {
					stopReason = StopLoopError
				}
				stopReasonSet = true
				return completedEpochs, lastLoss, err
			}
		}
	}

	// Late-epoch divergence guard — same contract as Fit.
	if minLossSet && lastLoss > minLoss && snapshot != nil {
		n.restoreWeights(snapshot)
		lastLoss = minLoss
	}
	return completedEpochs, lastLoss, nil
}
