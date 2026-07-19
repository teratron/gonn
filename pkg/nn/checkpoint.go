// Package nn — checkpoint integration for the Fit loop.
//
// WithCheckpoint arms periodic snapshot writes; Resume reconstructs a
// network from the newest snapshot in a directory. Snapshots are
// self-contained (config + weights + optimizer state), so Resume needs no
// separate config file (audit B6: pkg/checkpoint was fully implemented but
// nothing ever called it).
package nn

import (
	"github.com/teratron/gonn/pkg/checkpoint"
	"github.com/teratron/gonn/pkg/utils"
)

// WithCheckpoint arms periodic checkpointing: every everyN completed epochs
// Fit writes a self-contained snapshot into dir and applies the retention
// sweep. everyN == 0 is normalised to 1 (checkpoint every epoch). A zero
// SweepConfig selects checkpoint.DefaultSweepConfig (3 hot / 10 cold).
//
// Checkpoint write failures are logged at Warn and never abort training —
// a full disk should cost you the snapshot, not the run.
//
// AI-Meta:
//   - Purpose: Enable periodic training snapshots with retention during Fit.
//   - Usage: nn.New[float32](WithCheckpoint[float32]("ckpts", 10, checkpoint.SweepConfig{}), ...).
//   - Concurrency: Safe.
//   - Related: [Resume], [checkpoint.WriteSnapshot], [checkpoint.Sweep].
//   - Stability: Stable.
func WithCheckpoint[T utils.Float](dir string, everyN uint, sweep checkpoint.SweepConfig) Option[T] {
	return func(cfg *Config[T]) {
		cfg.CheckpointDir = dir
		if everyN == 0 {
			everyN = 1
		}
		cfg.CheckpointEvery = everyN
		cfg.CheckpointSweep = sweep
	}
}

// Resume reconstructs a network from the most recent snapshot in dir and
// returns it together with the epoch the snapshot was taken at. Weights and
// optimizer state are restored; note that the snapshot does not record the
// optimizer TYPE, so a run trained with a custom optimizer resumes on the
// default (SGD) and the state blob is skipped with a Warn if incompatible.
//
// The resumed network does not automatically continue checkpointing —
// re-arm with WithCheckpoint at construction time for managed runs, or
// call Save periodically.
//
// AI-Meta:
//   - Purpose: Restore a ready-to-train network from the newest checkpoint in a directory.
//   - Usage: n, epoch, err := nn.Resume[float32]("ckpts").
//   - Concurrency: SingleGoroutine.
//   - Errors: ErrIO (no snapshots, read failure), ErrIntegrity (corrupt snapshot), ErrUserConfig (schema mismatch).
//   - Related: [WithCheckpoint], [checkpoint.LoadLatest], [Load].
//   - Stability: Stable.
func Resume[T utils.Float](dir string) (*NN[T], uint64, error) {
	snap, path, err := checkpoint.LoadLatest[T](dir)
	if err != nil {
		return nil, 0, err
	}
	n, err := buildFromConfigDoc(snap.Config)
	if err != nil {
		return nil, 0, err
	}
	if err := installWeightsDoc(n, snap.Config, snap.Weights); err != nil {
		return nil, 0, err
	}
	doc := snap.Config
	n.persistDoc = &doc
	if len(snap.OptState) > 0 {
		if stateErr := n.opt.LoadState(snap.OptState); stateErr != nil {
			utils.Logger.Warn("Resume: optimizer state not restored (different optimizer type?)",
				"snapshot", path, "err", stateErr.Error())
		}
	}
	return n, snap.Iter, nil
}

// maybeWriteCheckpoint is Fit's per-epoch hook. It is a no-op unless
// WithCheckpoint armed a directory and the epoch lands on the everyN grid.
// All failures are logged at Warn — checkpointing is best-effort by design.
func (n *NN[T]) maybeWriteCheckpoint(epoch uint, mean, minLoss T, bestEpoch uint) {
	if n.cfg.CheckpointDir == "" || n.cfg.CheckpointEvery == 0 || epoch%n.cfg.CheckpointEvery != 0 {
		return
	}
	doc := n.persistenceDoc()
	snap := checkpoint.Snapshot[T]{
		Config:  doc,
		Weights: extractWeightsDoc(n, doc),
		Loss:    mean,
		Iter:    uint64(epoch),
		MinLossState: checkpoint.MinLossState[T]{
			Loss: minLoss,
			Iter: uint64(bestEpoch),
		},
		TopologyVersion: n.Network.TopologyVersion(),
	}
	if blob, err := n.opt.SaveState(); err == nil {
		snap.OptState = blob
	} else {
		utils.Logger.Warn("checkpoint: optimizer state not captured", "err", err.Error())
	}
	if _, err := checkpoint.WriteSnapshot(n.cfg.CheckpointDir, snap); err != nil {
		utils.Logger.Warn("checkpoint write failed",
			"dir", n.cfg.CheckpointDir, "epoch", epoch, "err", err.Error())
		return
	}
	if _, _, _, err := checkpoint.Sweep(n.cfg.CheckpointDir, n.cfg.CheckpointSweep); err != nil {
		utils.Logger.Warn("checkpoint sweep failed",
			"dir", n.cfg.CheckpointDir, "err", err.Error())
	}
}
