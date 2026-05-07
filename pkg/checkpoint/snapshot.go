// Package checkpoint — snapshot type and writer.
//
// Implements [l2-checkpointing-impl] §5.2 (snapshot struct) plus the
// atomic-write protocol from §5.3 — temp + Sync + Rename so a crash
// mid-write leaves either the prior snapshot intact or no snapshot at
// all (CHK-1).
//
// Stdlib only per C29: encoding/json, os, path/filepath, time.
package checkpoint

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/teratron/gonn/pkg/persistence"
	"github.com/teratron/gonn/pkg/utils"
)

// SchemaVersion is the wire-format version embedded in every Snapshot.
// A top-level schema_version field enables forward migration via the
// migrate() dispatcher in reader.go.
//
// AI-Meta:
//   - Purpose: Mark the on-disk snapshot format version for migration and compatibility checks.
//   - Related: [Snapshot], [LoadLatest].
const SchemaVersion = "1.0"

// MinLossState records the best loss observed so far and the iteration
// where it was reached; used by the training loop to roll back to the
// best-known state.
//
// AI-Meta:
//   - Purpose: Track the best-loss checkpoint for potential rollback during training.
//   - Related: [Snapshot].
type MinLossState[T utils.Float] struct {
	Iter uint64 `json:"iter"`
	Loss T      `json:"loss"`
}

// Snapshot is the complete resumable training state: config + weights
// documents (self-contained), iteration counter, RNG state, and rolling
// min-loss tracker.
//
// AI-Meta:
//   - Purpose: Self-contained training checkpoint; load via LoadLatest to resume a run.
//   - Related: [WriteSnapshot], [LoadLatest], [MinLossState], [persistence.ConfigDoc], [persistence.WeightsDoc].
type Snapshot[T utils.Float] struct {
	SchemaVersion string                    `json:"schema_version"`
	Iter          uint64                    `json:"iter"`
	Loss          T                         `json:"loss"`
	MinLossState  MinLossState[T]           `json:"min_loss_state"`
	RNGState      []byte                    `json:"rng_state,omitempty"`
	Config        persistence.ConfigDoc[T]  `json:"config"`
	Weights       persistence.WeightsDoc[T] `json:"weights"`
	Timestamp     int64                     `json:"timestamp"`
}

// snapshotName produces the canonical filename for a snapshot. The Iter
// and Timestamp pair sorts deterministically so retention sweep can
// keep the N most recent without parsing JSON.
func snapshotName(iter uint64, ts int64) string {
	return fmt.Sprintf("snap-%020d-%d.json", iter, ts)
}

// WriteSnapshot serialises snap to a new file inside dir using the atomic
// write protocol (tmp + Sync + Rename). SchemaVersion and Timestamp are
// auto-populated when callers leave them at their zero values.
//
// AI-Meta:
//   - Purpose: Persist a training snapshot atomically; returns the path of the written file.
//   - Errors: ErrIO (mkdir, create, write, sync, or rename failure).
//   - Related: [Snapshot], [LoadLatest], [Sweep].
func WriteSnapshot[T utils.Float](dir string, snap Snapshot[T]) (string, error) {
	if snap.SchemaVersion == "" {
		snap.SchemaVersion = SchemaVersion
	}
	if snap.Timestamp == 0 {
		snap.Timestamp = time.Now().Unix()
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", utils.Wrap(utils.ErrIO, err, "WriteSnapshot: mkdir %q", dir)
	}
	target := filepath.Join(dir, snapshotName(snap.Iter, snap.Timestamp))

	tmp, err := os.CreateTemp(dir, ".snap-*.json.tmp")
	if err != nil {
		return "", utils.Wrap(utils.ErrIO, err, "WriteSnapshot: create tmp")
	}
	tmpName := tmp.Name()

	if err := writeSnapshotBody(tmp, snap); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return "", err
	}
	if err := os.Rename(tmpName, target); err != nil {
		_ = os.Remove(tmpName)
		return "", utils.Wrap(utils.ErrIO, err, "WriteSnapshot: rename")
	}
	return target, nil
}

// writeSnapshotBody encodes snap into f and fsync+closes the handle.
// Pulled out so WriteSnapshot's atomic-write structure is one straight
// line of read-it-once steps with one cleanup site.
func writeSnapshotBody[T utils.Float](f *os.File, snap Snapshot[T]) error {
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(snap); err != nil {
		return utils.Wrap(utils.ErrIO, err, "WriteSnapshot: encode")
	}
	if err := f.Sync(); err != nil {
		return utils.Wrap(utils.ErrIO, err, "WriteSnapshot: sync")
	}
	if err := f.Close(); err != nil {
		return utils.Wrap(utils.ErrIO, err, "WriteSnapshot: close")
	}
	return nil
}
