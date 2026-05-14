package nn

import (
	"fmt"

	"github.com/teratron/gonn/pkg/utils"
)

// AndTrain continues training on a new sample set without rebuilding the
// network. Per l2-dataset-loader-impl §5.4 (FMT-6 / FMT-7 / FMT-8):
//
//   - Existing weights are preserved — only convergence counters are reset.
//   - Variadic Option overrides are applied to a snapshot of the config and
//     restored on return; the base configuration is left intact.
//   - The state machine must be Idle on entry; ErrNetworkRunning otherwise.
//   - Callbacks registered before the call remain active throughout; the
//     OnTrainEnd hook fires exactly once at the end of this AndTrain call.
//
// Typical use is pre-training then fine-tuning: Train on dataset A to
// initial convergence, then AndTrain on dataset B with a lower learning
// rate to refine — no weight reset, no callback re-registration.
//
// AI-Meta:
//   - Purpose: Continuation training on a new sample set with optional per-call option overrides.
//   - Usage: epochs, loss, err := n.AndTrain(samplesB, WithLearningRate[float32](0.01)).
//   - Concurrency: SingleGoroutine; must not run concurrently with Train, Fit, or another AndTrain.
//   - Errors: ErrUserConfig (network not Operational), ErrNetworkRunning (training already running), ErrInputData (empty samples).
//   - Related: [Fit], [Train], [Option], [utils.ErrNetworkRunning].
//   - Stability: Stable.
func (n *NN[T]) AndTrain(samples []Sample[T], opts ...Option[T]) (uint, T, error) {
	if n.stateField != stateOperational {
		return 0, 0, utils.Newf(utils.ErrUserConfig,
			"AndTrain: network is %s, must be Operational", n.stateField.String())
	}
	if state := n.control.Load(); state != controlIdle {
		return 0, 0, fmt.Errorf(
			"AndTrain: training lifecycle state must be Idle, got %d: %w",
			state, utils.ErrNetworkRunning)
	}
	if len(samples) == 0 {
		return 0, 0, utils.Newf(utils.ErrInputData,
			"AndTrain: samples must contain at least one entry")
	}

	// Snapshot the parts of NN[T] that opts may mutate so we can restore
	// the base configuration verbatim on return. Callbacks (n.callbacks)
	// are intentionally preserved across the call per FMT-7.
	originalCfg := n.cfg
	originalOpt := n.opt
	originalSched := n.sched
	originalReg := n.reg
	defer func() {
		n.cfg = originalCfg
		n.opt = originalOpt
		n.sched = originalSched
		n.reg = originalReg
	}()

	// Apply opts to the live config. We re-resolve runtime pointers below
	// so any optimizer / scheduler / regularizer override takes effect for
	// this call only.
	for _, o := range opts {
		if o != nil {
			o(&n.cfg)
		}
	}
	if n.cfg.Optimizer != nil {
		n.opt = n.cfg.Optimizer
	}
	if n.cfg.Scheduler != nil {
		n.sched = n.cfg.Scheduler
	}
	if n.cfg.Regularizer != nil {
		n.reg = n.cfg.Regularizer
	}

	// Fit owns the state-machine transitions (transitionToRunning /
	// transitionToIdle) and fires OnTrainEnd via its deferred cleanup —
	// no extra wiring needed here.
	return n.Fit(samples)
}
