package nn

import (
	"runtime"

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

// Pause requests a transition Running → Paused. The worker observes
// this at the next safe-point (between epochs) and blocks there until
// Resume or Stop is invoked.
//
// Returns ErrControl when called on a network that is not currently
// running — pause-while-idle is a state-machine misuse.
func (n *NN[T]) Pause() error {
	if !n.control.CompareAndSwap(controlRunning, controlPaused) {
		return utils.Newf(utils.ErrControl,
			"Pause: network is not currently training (state=%d)", n.control.Load())
	}
	utils.Logger.Debug("Pause requested")
	return nil
}

// Resume requests a transition Paused → Running. Idempotent on already-
// Running networks (returns nil); errors via ErrControl on Idle / Stopped.
func (n *NN[T]) Resume() error {
	if n.control.CompareAndSwap(controlPaused, controlRunning) {
		utils.Logger.Debug("Resume signalled")
		return nil
	}
	if n.control.Load() == controlRunning {
		return nil
	}
	return utils.Newf(utils.ErrControl,
		"Resume: network is not paused (state=%d)", n.control.Load())
}

// Stop requests an early exit from the training loop. The worker
// observes this at the next safe-point and returns from Fit. Idempotent
// on already-Stopped networks; allowed from Idle / Running / Paused.
func (n *NN[T]) Stop() error {
	for {
		cur := n.control.Load()
		if cur == controlStopped {
			return nil
		}
		if n.control.CompareAndSwap(cur, controlStopped) {
			utils.Logger.Debug("Stop signalled", "from", cur)
			return nil
		}
	}
}

// transitionToRunning is the worker-side counterpart to Pause/Resume —
// called by Fit at start. Moves Idle → Running via CAS so that a Stop
// request issued *before* Fit reaches this line is preserved (otherwise
// a slow goroutine startup would silently clobber the user's intent
// and the loop would run to MaxIterations).
func (n *NN[T]) transitionToRunning() {
	n.control.CompareAndSwap(controlIdle, controlRunning)
}

// transitionToIdle is invoked by Fit's deferred cleanup. Mirrors the
// pattern used in net/http: every Run() / Fit() must restore the Idle
// state so subsequent Train / Fit calls succeed.
func (n *NN[T]) transitionToIdle() {
	// Preserve Stopped flag if the user set it explicitly — leaves a
	// trace for tests that want to assert "this run was stopped". The
	// next Fit() call will reset to Running anyway.
	if n.control.Load() != controlStopped {
		n.control.Store(controlIdle)
	}
}

// awaitSafePoint is the inner loop's check that runs between epochs.
// Returns (stopped bool, err error):
//   - (false, nil) — keep training.
//   - (true, nil)  — Stop was requested; caller breaks out of the loop.
//   - (_, err)     — never returned in v0.1; reserved for future
//     timeout-based safe-points.
//
// While the state is Paused the goroutine yields via runtime.Gosched
// in a short busy-wait. The cost is acceptable because pauses are
// expected to be rare and short; a sync.Cond would add complexity for
// no visible gain on the typical training workload.
func (n *NN[T]) awaitSafePoint() (bool, error) {
	for {
		switch n.control.Load() {
		case controlRunning:
			return false, nil
		case controlStopped:
			return true, nil
		case controlPaused:
			runtime.Gosched()
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
