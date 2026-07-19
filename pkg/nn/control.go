package nn

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Lifecycle states for the training-control state machine. Stored as
// an int32 inside [NN.control] so transitions are atomic across the
// goroutine boundary between Train/Fit (the worker) and the user-facing
// Pause / Resume / Stop calls (the controller).
const (
	controlIdle    int32 = 0 // not training
	controlRunning int32 = 1 // training in progress
	controlPaused  int32 = 2 // training paused mid-epoch
	controlStopped int32 = 3 // stop requested; worker exits at next safe-point
)

// Pause requests a Running → Paused transition. The Fit goroutine observes
// this signal at the next safe-point (between epochs) and blocks until
// Resume or Stop is called.
//
// AI-Meta:
//   - Purpose: Temporarily suspend an in-progress Fit loop from another goroutine.
//   - Concurrency: Safe; uses atomic CAS and is designed for cross-goroutine calls.
//   - Errors: ErrControl (network is not currently Running).
//   - Related: [Resume], [Stop], [Fit].
//   - Stability: Stable.
func (n *NN[T]) Pause() error {
	if !n.control.CompareAndSwap(controlRunning, controlPaused) {
		return utils.Newf(utils.ErrControl,
			"Pause: network is not currently training (state=%d)", n.control.Load())
	}
	utils.Logger.Debug("Pause requested")
	return nil
}

// Resume requests a Paused → Running transition. Idempotent on already-Running
// networks (returns nil); errors via ErrControl on Idle or Stopped.
//
// AI-Meta:
//   - Purpose: Resume a Fit loop that was suspended by Pause.
//   - Concurrency: Safe; designed for cross-goroutine calls.
//   - Errors: ErrControl (network is not paused or running).
//   - Related: [Pause], [Stop], [Fit].
//   - Stability: Stable.
func (n *NN[T]) Resume() error {
	if n.control.CompareAndSwap(controlPaused, controlRunning) {
		n.wakePaused()
		utils.Logger.Debug("Resume signalled")
		return nil
	}
	if n.control.Load() == controlRunning {
		return nil
	}
	return utils.Newf(utils.ErrControl,
		"Resume: network is not paused (state=%d)", n.control.Load())
}

// Stop requests an early exit from the training loop. The Fit goroutine
// observes the signal at the next safe-point and returns. Idempotent on
// already-Stopped networks; valid from Idle, Running, or Paused.
//
// AI-Meta:
//   - Purpose: Signal Fit to terminate at the next epoch boundary; best-weight rollback still applies.
//   - Concurrency: Safe; designed for cross-goroutine calls.
//   - Errors: Never returns an error (idempotent CAS loop).
//   - Related: [Pause], [Resume], [Fit].
//   - Stability: Stable.
func (n *NN[T]) Stop() error {
	for {
		cur := n.control.Load()
		if cur == controlStopped {
			return nil
		}
		if n.control.CompareAndSwap(cur, controlStopped) {
			// A worker parked in awaitSafePoint's paused wait must observe
			// the transition immediately.
			n.wakePaused()
			utils.Logger.Debug("Stop signalled", "from", cur)
			return nil
		}
	}
}

// wakePaused broadcasts the pause condition under its mutex so a worker
// blocked in awaitSafePoint re-checks the control state. Broadcasting under
// pauseMu (not just after the CAS) closes the missed-wakeup window: the
// worker only Waits while holding pauseMu and after re-checking the state.
func (n *NN[T]) wakePaused() {
	if n.pauseCond == nil {
		return
	}
	n.pauseMu.Lock()
	n.pauseCond.Broadcast()
	n.pauseMu.Unlock()
}

// transitionToRunning is the worker-side counterpart to Pause/Resume —
// called by Fit at start. Moves Idle → Running via CAS so that a Stop
// request issued *before* Fit reaches this line is preserved (otherwise
// a slow goroutine startup would silently clobber the user's intent
// and the loop would run to MaxIterations).
func (n *NN[T]) transitionToRunning() {
	n.control.CompareAndSwap(controlIdle, controlRunning)
}

// transitionToIdle is invoked by Fit's deferred cleanup. Every Fit must restore
// the Idle state so subsequent Train / Fit / AndTrain calls succeed. Earlier
// code preserved a Stopped flag here, which left the network permanently sterile
// after the first Stop() — the next Fit saw Stopped at its first safe-point and
// exited with zero epochs (audit D5). The stop reason is already reported to
// OnTrainEnd callbacks, so no state trace is needed.
func (n *NN[T]) transitionToIdle() {
	n.control.Store(controlIdle)
}

// awaitSafePoint is the inner loop's check that runs between epochs.
// Returns (stopped bool, err error):
//   - (false, nil) — keep training.
//   - (true, nil)  — Stop was requested; caller breaks out of the loop.
//   - (_, err)     — never returned in v0.5; reserved for future
//     timeout-based safe-points.
//
// While the state is Paused the worker parks on pauseCond — zero CPU until
// Resume or Stop broadcasts (audit F: the historical runtime.Gosched loop
// burned a full core for the duration of every pause).
func (n *NN[T]) awaitSafePoint() (bool, error) {
	for {
		switch n.control.Load() {
		case controlRunning:
			return false, nil
		case controlStopped:
			return true, nil
		case controlPaused:
			if n.pauseCond == nil {
				// Zero-value NN outside the builder path — nothing can ever
				// broadcast, so do not park; treat as still running.
				return false, nil
			}
			n.pauseMu.Lock()
			for n.control.Load() == controlPaused {
				n.pauseCond.Wait()
			}
			n.pauseMu.Unlock()
			continue
		case controlIdle:
			// Worker entered the loop without transitionToRunning —
			// programmer error in Fit. Fall back to Running so the
			// loop makes progress instead of spinning indefinitely.
			n.control.Store(controlRunning)
			return false, nil
		default:
			return false, utils.Newf(utils.ErrControl,
				"awaitSafePoint: unknown state %d", n.control.Load())
		}
	}
}
