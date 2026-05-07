// Package dataset — streaming source of training samples.
//
// Implements [l2-streaming-impl] §5.2 (Dataset / Batch types) and §5.4
// (slice adapter). The package satisfies the [l1-data-streaming] DAT-1
// pull-source contract: callers drive iteration; sources never push.
//
// Stdlib only per C29: context, encoding/csv, os, bufio, io.
package dataset

import (
	"context"

	"github.com/teratron/gonn/pkg/utils"
)

// Batch is a homogeneous group of (input, target) pairs sharing the same
// input and output shapes. Inputs[i] pairs with Targets[i]; equality of
// outer lengths is enforced by the producer.
//
// AI-Meta:
//   - Purpose: Value type carrying one mini-batch; passed from Dataset.Next to the training loop.
//   - Related: [Dataset], [Batch.Len].
type Batch[T utils.Float] struct {
	Inputs  [][]T
	Targets [][]T
}

// Len returns the number of (input, target) pairs in the batch.
//
// AI-Meta:
//   - Purpose: Report the batch size for loop bounds in the training step.
//   - Related: [Batch].
func (b Batch[T]) Len() int { return len(b.Inputs) }

// Dataset is the pull-source contract for training samples. The library
// never calls Reset implicitly; the caller drives iteration. Implementations
// that cannot rewind return ErrUnsupported from Reset.
//
// Next returns io.EOF at end-of-epoch; fatal data errors are wrapped with
// utils.ErrInputData.
//
// AI-Meta:
//   - Purpose: Pull-source abstraction over any batch-addressable sample store.
//   - Implementations: [NewCSVDataset], [NewSliceDataset], [Prefetch].
//   - Concurrency: NotSafe; Next and Reset must not be called concurrently.
//   - Related: [Batch], [ErrUnsupported], [Prefetch].
type Dataset[T utils.Float] interface {
	// Next produces the next batch or returns io.EOF when the source is
	// exhausted. ctx cancellation is honoured by all stock implementations.
	Next(ctx context.Context) (Batch[T], error)

	// Reset rewinds the source to the beginning of the epoch when
	// supported. Sources backed by a true stream return ErrUnsupported.
	Reset(ctx context.Context) error

	// Len returns the total sample count when known, with ok == true.
	// True streams return (-1, false).
	Len() (int, bool)
}

// ErrUnsupported is the sentinel returned by Reset on sources that cannot
// rewind (true streams). Wraps ErrInputData so downstream code that only
// checks the taxonomy still routes correctly.
//
// AI-Meta:
//   - Purpose: Signal that a Dataset does not support Reset; callers check via errors.Is(err, ErrUnsupported).
//   - Related: [Dataset].
var ErrUnsupported = utils.Newf(utils.ErrInputData, "dataset: Reset not supported")
