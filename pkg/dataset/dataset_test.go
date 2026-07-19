// Package dataset — round-trip and error-path tests.
//
// Covers DAT-1 (pull semantics), DAT-2 (homogeneous batch), DAT-3
// (memory-bounded prefetch), DAT-4 (errors wrapped with ErrInputData).
package dataset

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/teratron/gonn/pkg/utils"
)

// drainAll reads ds.Next until EOF and returns the concatenated samples
// or the first non-EOF error encountered.
func drainAll[T utils.Float](t *testing.T, ds Dataset[T]) ([][]T, [][]T, error) {
	t.Helper()
	var ins, tgts [][]T
	ctx := context.Background()
	for {
		b, err := ds.Next(ctx)
		if err == io.EOF {
			return ins, tgts, nil
		}
		if err != nil {
			return ins, tgts, err
		}
		ins = append(ins, b.Inputs...)
		tgts = append(tgts, b.Targets...)
	}
}

func TestSliceDatasetIteration(t *testing.T) {
	inputs := [][]float32{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	targets := [][]float32{{1}, {0}, {1}, {0}}
	ds, err := NewSliceDataset(inputs, targets, 2)
	if err != nil {
		t.Fatalf("NewSliceDataset: %v", err)
	}
	if n, ok := ds.Len(); !ok || n != 4 {
		t.Errorf("Len = (%d, %v), want (4, true)", n, ok)
	}
	gotIn, gotT, err := drainAll(t, ds)
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	if len(gotIn) != 4 || len(gotT) != 4 {
		t.Errorf("drained %d samples, want 4", len(gotIn))
	}
	if gotIn[2][0] != 5 {
		t.Errorf("sample[2][0] = %v, want 5", gotIn[2][0])
	}
}

func TestSliceDatasetReset(t *testing.T) {
	ds, _ := NewSliceDataset(
		[][]float32{{1}, {2}}, [][]float32{{1}, {2}}, 1)
	if _, _, err := drainAll(t, ds); err != nil {
		t.Fatal(err)
	}
	if err := ds.Reset(context.Background()); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	in, _, err := drainAll(t, ds)
	if err != nil {
		t.Fatal(err)
	}
	if len(in) != 2 {
		t.Errorf("after Reset drained %d, want 2", len(in))
	}
}

func TestSliceDatasetPartialFinalBatch(t *testing.T) {
	ds, _ := NewSliceDataset(
		[][]float32{{1}, {2}, {3}}, [][]float32{{1}, {2}, {3}}, 2)
	b1, err := ds.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if b1.Len() != 2 {
		t.Errorf("first batch len = %d, want 2", b1.Len())
	}
	b2, err := ds.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if b2.Len() != 1 {
		t.Errorf("tail batch len = %d, want 1", b2.Len())
	}
	if _, err := ds.Next(context.Background()); err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
}

func TestSliceDatasetValidation(t *testing.T) {
	if _, err := NewSliceDataset([][]float32{{1}}, [][]float32{}, 1); err == nil {
		t.Error("expected error on len mismatch")
	}
	if _, err := NewSliceDataset([][]float32{{1}}, [][]float32{{1}}, 0); err == nil {
		t.Error("expected error on zero batch")
	}
}

func TestSliceDatasetContextCancel(t *testing.T) {
	ds, _ := NewSliceDataset([][]float32{{1}}, [][]float32{{1}}, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ds.Next(ctx); err == nil {
		t.Error("expected ctx error")
	}
	if err := ds.Reset(ctx); err == nil {
		t.Error("expected ctx error on Reset")
	}
}

func TestCSVDatasetHappyPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")
	body := "0.1,0.2,1\n0.3,0.4,0\n0.5,0.6,1\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	ds, err := NewCSVDataset[float32](path, 2, 1, 2)
	if err != nil {
		t.Fatalf("NewCSVDataset: %v", err)
	}
	if n, ok := ds.Len(); ok || n != -1 {
		t.Errorf("Len = (%d, %v), want (-1, false)", n, ok)
	}
	in, tgt, err := drainAll(t, ds)
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	if len(in) != 3 {
		t.Fatalf("drained %d rows, want 3", len(in))
	}
	if in[1][1] != float32(0.4) {
		t.Errorf("row 1 col 1 = %v, want 0.4", in[1][1])
	}
	if tgt[2][0] != float32(1) {
		t.Errorf("row 2 target = %v, want 1", tgt[2][0])
	}
}

func TestCSVDatasetMalformedRow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.csv")
	body := "0.1,0.2,1\nbroken,row,here\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	ds, err := NewCSVDataset[float32](path, 2, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	// First row is fine.
	if _, err := ds.Next(context.Background()); err != nil {
		t.Fatalf("first row: %v", err)
	}
	// Second row has non-numeric cell.
	_, err = ds.Next(context.Background())
	if err == nil || !errors.Is(err, utils.ErrInputData) {
		t.Errorf("expected ErrInputData, got %v", err)
	}
}

func TestCSVDatasetReset(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")
	body := "1,2,1\n3,4,0\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	ds, err := NewCSVDataset[float64](path, 2, 1, 10)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := drainAll(t, ds); err != nil {
		t.Fatal(err)
	}
	if err := ds.Reset(context.Background()); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	in, _, err := drainAll(t, ds)
	if err != nil {
		t.Fatal(err)
	}
	if len(in) != 2 {
		t.Errorf("after Reset drained %d, want 2", len(in))
	}
}

func TestCSVDatasetMissingFile(t *testing.T) {
	_, err := NewCSVDataset[float32]("/no/such/path.csv", 1, 1, 1)
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("expected ErrIO, got %v", err)
	}
}

func TestCSVDatasetValidation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.csv")
	_ = os.WriteFile(path, []byte("1,1\n"), 0o644)
	for _, tc := range []struct {
		name                       string
		inSize, outSize, batchSize int
	}{
		{"zero-input", 0, 1, 1},
		{"zero-output", 1, 0, 1},
		{"zero-batch", 1, 1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewCSVDataset[float32](path, tc.inSize, tc.outSize, tc.batchSize)
			if err == nil || !errors.Is(err, utils.ErrUserConfig) {
				t.Errorf("expected ErrUserConfig, got %v", err)
			}
		})
	}
}

func TestPrefetchOrdering(t *testing.T) {
	inputs := [][]float32{{1}, {2}, {3}, {4}, {5}}
	targets := [][]float32{{1}, {2}, {3}, {4}, {5}}
	inner, _ := NewSliceDataset(inputs, targets, 1)
	ds, err := Prefetch(inner, 2)
	if err != nil {
		t.Fatal(err)
	}
	in, _, err := drainAll(t, ds)
	if err != nil {
		t.Fatal(err)
	}
	if len(in) != 5 {
		t.Fatalf("drained %d, want 5", len(in))
	}
	for i, v := range in {
		if v[0] != float32(i+1) {
			t.Errorf("order broken at %d: got %v", i, v[0])
		}
	}
}

func TestPrefetchBackpressure(t *testing.T) {
	// Counter increments each time inner.Next is called.
	var counter atomic.Int32
	inner := &countingDataset{counter: &counter, total: 100}

	ds, err := Prefetch(inner, 1) // capacity = 2
	if err != nil {
		t.Fatal(err)
	}

	// Sleep to let producer fill the channel; then check it stalled at
	// capacity rather than producing all 100 rows.
	time.Sleep(50 * time.Millisecond)
	produced := counter.Load()
	if produced > 4 {
		t.Errorf("backpressure violated: producer made %d batches before consumer drained any (cap=2)", produced)
	}

	in, _, err := drainAll(t, ds)
	if err != nil {
		t.Fatal(err)
	}
	if len(in) != 100 {
		t.Errorf("drained %d, want 100", len(in))
	}
}

// countingDataset emits sequential singleton batches up to total samples
// while incrementing counter on each Next call.
type countingDataset struct {
	counter *atomic.Int32
	total   int
	cursor  int
}

func (d *countingDataset) Next(ctx context.Context) (Batch[float32], error) {
	if err := ctx.Err(); err != nil {
		return Batch[float32]{}, err
	}
	d.counter.Add(1)
	if d.cursor >= d.total {
		return Batch[float32]{}, io.EOF
	}
	b := Batch[float32]{
		Inputs:  [][]float32{{float32(d.cursor)}},
		Targets: [][]float32{{float32(d.cursor)}},
	}
	d.cursor++
	return b, nil
}

func (d *countingDataset) Reset(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	d.cursor = 0
	return nil
}

func (d *countingDataset) Len() (int, bool) { return d.total, true }

func TestPrefetchValidation(t *testing.T) {
	if _, err := Prefetch[float32](nil, 1); err == nil {
		t.Error("expected error on nil inner")
	}
	inner, _ := NewSliceDataset([][]float32{{1}}, [][]float32{{1}}, 1)
	if _, err := Prefetch(inner, -1); err == nil {
		t.Error("expected error on negative prefetch")
	}
}

func TestPrefetchContextCancel(t *testing.T) {
	inner := &countingDataset{counter: &atomic.Int32{}, total: 1_000_000}
	ds, _ := Prefetch(inner, 0)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ds.Next(ctx)
	if err == nil {
		t.Error("expected error after cancel")
	}
}

func TestPrefetchReset(t *testing.T) {
	inputs := [][]float32{{1}, {2}}
	targets := [][]float32{{1}, {2}}
	inner, _ := NewSliceDataset(inputs, targets, 1)
	ds, _ := Prefetch(inner, 0)
	if _, _, err := drainAll(t, ds); err != nil {
		t.Fatal(err)
	}
	if err := ds.Reset(context.Background()); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	in, _, err := drainAll(t, ds)
	if err != nil {
		t.Fatal(err)
	}
	if len(in) != 2 {
		t.Errorf("after Reset drained %d, want 2", len(in))
	}
}

func TestPrefetchPropagatesError(t *testing.T) {
	inner := &errorDataset{err: utils.Newf(utils.ErrInputData, "synthetic")}
	ds, _ := Prefetch(inner, 1)
	_, err := ds.Next(context.Background())
	if err == nil || !errors.Is(err, utils.ErrInputData) {
		t.Errorf("expected ErrInputData propagation, got %v", err)
	}
}

type errorDataset struct{ err error }

func (e *errorDataset) Next(ctx context.Context) (Batch[float32], error) {
	return Batch[float32]{}, e.err
}
func (e *errorDataset) Reset(ctx context.Context) error { return ErrUnsupported }
func (e *errorDataset) Len() (int, bool)                { return 0, false }

func TestErrUnsupportedMessage(t *testing.T) {
	if !strings.Contains(ErrUnsupported.Error(), "Reset") {
		t.Errorf("expected Reset in ErrUnsupported, got %q", ErrUnsupported.Error())
	}
}
