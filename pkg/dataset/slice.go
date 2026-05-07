// Package dataset — in-memory slice adapter.
//
// Implements [l2-streaming-impl] §5.4: NewSliceDataset wraps already-resident
// [][]T arrays into the Dataset interface so existing code paths can migrate
// to batched streaming without changing their data source first.
package dataset

import (
	"context"
	"io"

	"github.com/teratron/gonn/pkg/utils"
)

// NewSliceDataset wraps in-memory slices into a Dataset[T]. inputs[i] pairs
// with targets[i]; both slices must have equal outer length. batchSize must
// be positive. The returned dataset supports Reset and reports a known Len.
//
// Intended for unit tests and for migrating single-sample Train call sites.
//
// AI-Meta:
//   - Purpose: Wrap resident [][]T arrays into the Dataset interface for batch-streaming without I/O.
//   - Usage: ds, err := dataset.NewSliceDataset[float32](inputs, targets, 16).
//   - Errors: ErrUserConfig (mismatched slice lengths, non-positive batchSize).
//   - Related: [Dataset], [NewCSVDataset], [Prefetch].
func NewSliceDataset[T utils.Float](inputs, targets [][]T, batchSize int) (Dataset[T], error) {
	if len(inputs) != len(targets) {
		return nil, utils.Newf(utils.ErrUserConfig,
			"NewSliceDataset: inputs len %d != targets len %d", len(inputs), len(targets))
	}
	if batchSize <= 0 {
		return nil, utils.NewSizeError("NewSliceDataset.batchSize", batchSize, "positive")
	}
	return &sliceDataset[T]{
		inputs:  inputs,
		targets: targets,
		batch:   batchSize,
	}, nil
}

type sliceDataset[T utils.Float] struct {
	inputs  [][]T
	targets [][]T
	batch   int
	cursor  int
}

func (s *sliceDataset[T]) Next(ctx context.Context) (Batch[T], error) {
	if err := ctx.Err(); err != nil {
		return Batch[T]{}, err
	}
	if s.cursor >= len(s.inputs) {
		return Batch[T]{}, io.EOF
	}
	end := min(s.cursor+s.batch, len(s.inputs))
	batch := Batch[T]{
		Inputs:  s.inputs[s.cursor:end],
		Targets: s.targets[s.cursor:end],
	}
	s.cursor = end
	return batch, nil
}

func (s *sliceDataset[T]) Reset(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.cursor = 0
	return nil
}

func (s *sliceDataset[T]) Len() (int, bool) { return len(s.inputs), true }
