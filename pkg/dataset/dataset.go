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

// Batch is a homogeneous group of (input, target) pairs all sharing the
// same input and output shapes (DAT-2). Inputs[i] is paired with
// Targets[i]; len(Inputs) == len(Targets) is enforced by the producer.
type Batch[T utils.Float] struct {
	Inputs  [][]T
	Targets [][]T
}

// Len returns the batch size — number of (input, target) pairs.
func (b Batch[T]) Len() int { return len(b.Inputs) }

// Dataset is the pull-source contract over training samples (DAT-1). The
// library never calls Reset implicitly and never seeks; the caller drives
// the iteration. Implementations that cannot rewind should return
// ErrUnsupported from Reset.
//
// Next returns io.EOF at end-of-epoch — distinct from a fatal data error
// which is wrapped with utils.ErrInputData per DAT-4.
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
// rewind (true streams). Callers route via errors.Is — the value wraps
// ErrInputData so downstream code that only checks the taxonomy still
// works correctly.
var ErrUnsupported = utils.Newf(utils.ErrInputData, "dataset: Reset not supported")
