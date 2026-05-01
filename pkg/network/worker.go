// Package network — bounded worker pool stub.
//
// Implements [l2-perf-impl] §5.3 (PERF-3): a worker pool sized to
// runtime.GOMAXPROCS(0) for batch-level parallelism. The Phase-3 stub
// exposes the constructor and bookkeeping fields; per-batch dispatch
// is wired in once the training loop adopts batches (post-MVP).
package network

import (
	"runtime"
	"sync"
)

// WorkerPool dispatches independent jobs to up to Workers goroutines.
// Construction is cheap; jobs is unbuffered so closing it acts as the
// shutdown signal.
type WorkerPool struct {
	Workers int
	jobs    chan func()
	wg      sync.WaitGroup
	once    sync.Once
}

// NewWorkerPool creates a pool sized to GOMAXPROCS — the Go runtime
// already obeys the OS thread cap, so using GOMAXPROCS keeps NN
// parallelism bounded by the same dial users already turn (PERF-3).
func NewWorkerPool() *WorkerPool {
	workers := max(runtime.GOMAXPROCS(0), 1)
	p := &WorkerPool{
		Workers: workers,
		jobs:    make(chan func()),
	}
	p.wg.Add(workers)
	for range workers {
		go p.run()
	}
	return p
}

// run is the per-worker loop. It exits when jobs is closed.
func (p *WorkerPool) run() {
	defer p.wg.Done()
	for job := range p.jobs {
		job()
	}
}

// Submit enqueues fn for execution by the next available worker. The
// call blocks if all workers are busy — natural backpressure.
func (p *WorkerPool) Submit(fn func()) {
	p.jobs <- fn
}

// Stop closes the jobs channel and waits for the workers to drain.
// Idempotent — multiple Stop calls are safe.
func (p *WorkerPool) Stop() {
	p.once.Do(func() {
		close(p.jobs)
	})
	p.wg.Wait()
}
