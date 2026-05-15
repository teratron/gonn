// Package dataset — prefetch decorator.
//
// Implements [l2-streaming-impl] §5.3: a single-producer goroutine pulls
// batches ahead of the consumer and parks them in a bounded channel of
// capacity prefetch+1. This satisfies DAT-3 (memory bound) and gives the
// trainer natural backpressure — the producer blocks on a full channel.
package dataset

import (
	"context"
	"errors"
	"io"
	"sync"

	"github.com/teratron/gonn/pkg/utils"
)

// Prefetch wraps inner with a background producer goroutine. The channel
// buffer holds at most prefetch+1 batches, bounding residency to
// (prefetch+1) × batchSize samples. A value of 0 still buffers one batch
// ahead. Reset rewinds inner and restarts the producer.
//
// AI-Meta:
//   - Purpose: Overlap I/O with training by pulling batches ahead of the consumer.
//   - Usage: ds, err := dataset.Prefetch[float32](inner, 2).
//   - Errors: ErrUserConfig (nil inner, negative prefetch).
//   - Concurrency: Safe; consumer calls Next/Reset, producer runs in a separate goroutine.
//   - Related: [Dataset], [NewCSVDataset], [NewSliceDataset].
func Prefetch[T utils.Float](inner Dataset[T], prefetch int) (Dataset[T], error) {
	if inner == nil {
		return nil, utils.Newf(utils.ErrUserConfig, "Prefetch: inner dataset is nil")
	}
	if prefetch < 0 {
		return nil, utils.NewSizeError("Prefetch.prefetch", prefetch, "non-negative")
	}
	p := &prefetchDataset[T]{
		inner:    inner,
		capacity: prefetch + 1,
	}
	p.start()
	return p, nil
}

type prefetchResult[T utils.Float] struct {
	err   error
	batch Batch[T]
}

type prefetchDataset[T utils.Float] struct {
	inner    Dataset[T]
	ch       chan prefetchResult[T]
	cancel   context.CancelFunc
	wg       sync.WaitGroup
	capacity int
	mu       sync.Mutex
}

// start spawns the producer goroutine. Called from the constructor and
// from Reset. The caller holds p.mu while invoking start so no observer
// can race a partially-initialised state.
func (p *prefetchDataset[T]) start() {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan prefetchResult[T], p.capacity)
	p.ch = ch
	p.cancel = cancel
	p.wg.Add(1)
	go p.run(ctx, ch)
}

// run is the producer body: pull from inner, push to ch until ctx is
// cancelled or inner returns io.EOF / fatal error. The terminal sentinel
// (io.EOF or wrapped error) is delivered to the consumer before the
// channel is closed, so the consumer always observes the cause of
// termination.
func (p *prefetchDataset[T]) run(ctx context.Context, ch chan prefetchResult[T]) {
	defer p.wg.Done()
	defer close(ch)
	for {
		batch, err := p.inner.Next(ctx)
		select {
		case ch <- prefetchResult[T]{batch: batch, err: err}:
		case <-ctx.Done():
			return
		}
		if err != nil {
			return
		}
	}
}

// Next blocks until a prefetched batch is available or ctx is cancelled.
// Closed-channel reads return io.EOF so the consumer sees a clean epoch
// boundary even if the producer terminated early.
func (p *prefetchDataset[T]) Next(ctx context.Context) (Batch[T], error) {
	if err := ctx.Err(); err != nil {
		return Batch[T]{}, err
	}
	p.mu.Lock()
	ch := p.ch
	p.mu.Unlock()
	select {
	case r, ok := <-ch:
		if !ok {
			return Batch[T]{}, io.EOF
		}
		return r.batch, r.err
	case <-ctx.Done():
		return Batch[T]{}, ctx.Err()
	}
}

// Reset stops the current producer, rewinds inner, and restarts the
// goroutine. Sources that do not support Reset propagate ErrUnsupported
// and the prefetch wrapper does not restart the producer.
func (p *prefetchDataset[T]) Reset(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cancel != nil {
		p.cancel()
	}
	p.wg.Wait()
	if err := p.inner.Reset(ctx); err != nil {
		if errors.Is(err, ErrUnsupported) {
			return ErrUnsupported
		}
		return err
	}
	p.start()
	return nil
}

// Len delegates to the inner dataset; prefetching does not change the
// epoch length.
func (p *prefetchDataset[T]) Len() (int, bool) { return p.inner.Len() }
