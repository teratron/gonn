// Package checkpoint — tests covering atomic write, gzip round-trip,
// retention, and schema migration.
package checkpoint

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/teratron/gonn/pkg/persistence"
	"github.com/teratron/gonn/pkg/utils"
)

func sampleSnapshot() Snapshot[float32] {
	cfg := persistence.ConfigDoc[float32]{
		SchemaVersion: persistence.SchemaVersion,
		Precision:     "float32",
		InputSize:     2,
		Output:        persistence.OutputDoc{Size: 1, Activation: "Sigmoid", Bias: true},
		Training: persistence.TrainingDoc[float32]{
			LearningRate: 0.3, Loss: "MSE", LossLimit: 1e-4, MaxIterations: 1000, WeightInit: "xavier",
		},
	}
	return Snapshot[float32]{
		Iter:         42,
		Loss:         0.123,
		MinLossState: MinLossState[float32]{Iter: 40, Loss: 0.1},
		RNGState:     []byte{1, 2, 3, 4},
		Config:       cfg,
		Weights: persistence.WeightsDoc[float32]{
			SchemaVersion: persistence.SchemaVersion,
			Layers: []persistence.LayerWeights[float32]{
				{Name: "out", Weights: [][]float32{{0.1, 0.2}}, Biases: []float32{0.05}},
			},
		},
	}
}

func TestSnapshotAtomicWriteAndLoad(t *testing.T) {
	dir := t.TempDir()
	snap := sampleSnapshot()

	path, err := WriteSnapshot(dir, snap)
	if err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}
	if !strings.HasSuffix(path, ".json") {
		t.Errorf("path = %q, want .json suffix", path)
	}

	got, fromPath, err := LoadLatest[float32](dir)
	if err != nil {
		t.Fatalf("LoadLatest: %v", err)
	}
	if fromPath != path {
		t.Errorf("loaded from %q, wrote to %q", fromPath, path)
	}
	if got.Iter != snap.Iter {
		t.Errorf("Iter = %d, want %d", got.Iter, snap.Iter)
	}
	if got.RNGState[0] != 1 || got.RNGState[3] != 4 {
		t.Errorf("RNGState lost: %v", got.RNGState)
	}
}

func TestLoadLatestEmptyDir(t *testing.T) {
	_, _, err := LoadLatest[float32](t.TempDir())
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestLoadLatestMissingDir(t *testing.T) {
	_, _, err := LoadLatest[float32](filepath.Join(t.TempDir(), "no-such-dir"))
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestLoadLatestTruncatedJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap-00000000000000000001-100.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := LoadLatest[float32](dir)
	if err == nil || !errors.Is(err, utils.ErrIntegrity) {
		t.Errorf("expected ErrIntegrity, got %v", err)
	}
}

func TestLoadLatestPicksHighestIter(t *testing.T) {
	dir := t.TempDir()
	for _, iter := range []uint64{1, 5, 3, 12, 7} {
		snap := sampleSnapshot()
		snap.Iter = iter
		snap.Timestamp = time.Now().Unix() + int64(iter) // stable ordering
		if _, err := WriteSnapshot(dir, snap); err != nil {
			t.Fatal(err)
		}
	}
	got, _, err := LoadLatest[float32](dir)
	if err != nil {
		t.Fatal(err)
	}
	if got.Iter != 12 {
		t.Errorf("Iter = %d, want 12", got.Iter)
	}
}

func TestGzipRoundTrip(t *testing.T) {
	dir := t.TempDir()
	snap := sampleSnapshot()
	path, err := WriteSnapshot(dir, snap)
	if err != nil {
		t.Fatal(err)
	}
	gzPath, err := gzipFile(path)
	if err != nil {
		t.Fatalf("gzipFile: %v", err)
	}
	if !strings.HasSuffix(gzPath, ".gz") {
		t.Errorf("gz path = %q", gzPath)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("source not removed: %v", err)
	}
	got, _, err := LoadLatest[float32](dir)
	if err != nil {
		t.Fatalf("LoadLatest after gzip: %v", err)
	}
	if got.Iter != snap.Iter {
		t.Errorf("Iter = %d, want %d", got.Iter, snap.Iter)
	}
}

func TestGzipMissingSource(t *testing.T) {
	_, err := gzipFile(filepath.Join(t.TempDir(), "no-such.json"))
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestSweepRetention(t *testing.T) {
	dir := t.TempDir()
	for i := uint64(1); i <= 8; i++ {
		snap := sampleSnapshot()
		snap.Iter = i
		snap.Timestamp = int64(i)
		if _, err := WriteSnapshot(dir, snap); err != nil {
			t.Fatal(err)
		}
	}
	hot, cold, deleted, err := Sweep(dir, SweepConfig{HotN: 2, ColdM: 3})
	if err != nil {
		t.Fatalf("Sweep: %v", err)
	}
	if hot != 2 || cold != 3 || deleted != 3 {
		t.Errorf("hot=%d cold=%d deleted=%d, want 2/3/3", hot, cold, deleted)
	}
	// Snapshot 8 (newest) must still be the loadable head.
	got, _, err := LoadLatest[float32](dir)
	if err != nil {
		t.Fatalf("LoadLatest: %v", err)
	}
	if got.Iter != 8 {
		t.Errorf("Iter = %d, want 8", got.Iter)
	}
}

func TestSweepDefaultConfig(t *testing.T) {
	dir := t.TempDir()
	snap := sampleSnapshot()
	if _, err := WriteSnapshot(dir, snap); err != nil {
		t.Fatal(err)
	}
	hot, cold, deleted, err := Sweep(dir, SweepConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if hot+cold+deleted != 1 {
		t.Errorf("expected 1 file accounted for, got hot=%d cold=%d del=%d", hot, cold, deleted)
	}
}

func TestSweepIgnoresUnrelatedFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "subdir"), 0o755); err != nil {
		t.Fatal(err)
	}
	hot, cold, deleted, err := Sweep(dir, DefaultSweepConfig)
	if err != nil {
		t.Fatal(err)
	}
	if hot+cold+deleted != 0 {
		t.Errorf("touched unrelated files: hot=%d cold=%d del=%d", hot, cold, deleted)
	}
}

func TestSweepMissingDir(t *testing.T) {
	_, _, _, err := Sweep(filepath.Join(t.TempDir(), "nope"), DefaultSweepConfig)
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestSweeperGoroutineLifecycle(t *testing.T) {
	dir := t.TempDir()
	for i := uint64(1); i <= 5; i++ {
		snap := sampleSnapshot()
		snap.Iter = i
		snap.Timestamp = int64(i)
		if _, err := WriteSnapshot(dir, snap); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s := StartSweeper(ctx, dir, SweepConfig{HotN: 1, ColdM: 1}, 5*time.Millisecond)
	time.Sleep(40 * time.Millisecond)
	s.Stop()
	s.Stop() // idempotent

	entries, _ := os.ReadDir(dir)
	if len(entries) > 5 {
		t.Errorf("sweeper did not run: %d files", len(entries))
	}
}

func TestSweeperContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	s := StartSweeper(ctx, t.TempDir(), DefaultSweepConfig, 5*time.Millisecond)
	cancel()
	s.Stop()
}

func TestMigrateUnknownSchema(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap-00000000000000000001-100.json")
	body := `{"schema_version":"99.0","iter":1,"loss":0,"min_loss_state":{"iter":0,"loss":0},"config":{"schema_version":"1.0.0","precision":"float32","input_size":2,"hidden_layers":[],"output":{"size":1,"activation":"Sigmoid","bias":true},"training":{"learning_rate":0.3,"loss":"MSE","loss_limit":0.0001,"max_iterations":1000,"weight_init":"xavier"}},"weights":{"schema_version":"1.0.0","config_hash":"sha256:0","layers":[]},"timestamp":1}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := LoadLatest[float32](dir)
	if err == nil || !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
}

func TestSnapshotWriteCreatesDir(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "deep")
	if _, err := WriteSnapshot(dir, sampleSnapshot()); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Errorf("directory not created: %v", err)
	}
}

func TestJoinErrorsSinglePassThrough(t *testing.T) {
	one := errors.New("one")
	if got := joinErrors([]error{one}); got != one {
		t.Errorf("singleton not passed through: %v", got)
	}
	if got := joinErrors(nil); got != nil {
		t.Errorf("empty must yield nil, got %v", got)
	}
	if got := joinErrors([]error{one, errors.New("two")}); got == nil {
		t.Error("expected combined error, got nil")
	}
}

func TestWriteSnapshotEncodeFailureOnNaN(t *testing.T) {
	dir := t.TempDir()
	snap := sampleSnapshot()
	// json.Marshal rejects NaN / +Inf for floats — verifies the encode
	// error path in WriteSnapshot.
	snap.Loss = float32(nanLoss())
	_, err := WriteSnapshot(dir, snap)
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO from NaN encode, got %v", err)
	}
}

func nanLoss() float64 {
	zero := float64(0)
	return zero / zero
}

func TestWriteSnapshotRenameFailureWhenTargetIsDir(t *testing.T) {
	dir := t.TempDir()
	snap := sampleSnapshot()
	snap.Iter = 99
	snap.Timestamp = 12345
	target := filepath.Join(dir, snapshotName(snap.Iter, snap.Timestamp))
	// Pre-create target as a non-empty directory; Rename must fail.
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(target, "blocker"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := WriteSnapshot(dir, snap)
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO on rename failure, got %v", err)
	}
}

func TestLoadLatestCorruptedGzip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap-00000000000000000001-100.json.gz")
	// gzip magic bytes followed by garbage — passes the magic check but
	// fails the gzip stream parse.
	body := []byte{0x1f, 0x8b, 0x08, 0x00, 0xff, 0xff, 0xff}
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := LoadLatest[float32](dir)
	if err == nil || !errors.Is(err, utils.ErrIntegrity) {
		t.Errorf("expected ErrIntegrity from broken gzip, got %v", err)
	}
}

func TestWriteSnapshotMkdirFailure(t *testing.T) {
	// Create a regular file then try to write a snapshot into a path
	// that treats that file as the parent directory — MkdirAll fails.
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(blocker, "snapdir")
	_, err := WriteSnapshot(target, sampleSnapshot())
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestSnapshotSchemaDefaults(t *testing.T) {
	dir := t.TempDir()
	snap := sampleSnapshot()
	snap.SchemaVersion = ""
	snap.Timestamp = 0
	path, err := WriteSnapshot(dir, snap)
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := LoadLatest[float32](dir)
	if err != nil {
		t.Fatal(err)
	}
	if got.SchemaVersion != SchemaVersion {
		t.Errorf("SchemaVersion = %q, want %q", got.SchemaVersion, SchemaVersion)
	}
	if got.Timestamp == 0 {
		t.Error("Timestamp not auto-populated")
	}
	_ = path
}
