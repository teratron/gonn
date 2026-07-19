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
// The jobs channel is unbuffered — closing it during Stop acts as the
// shutdown signal and drains all in-flight work.
//
// AI-Meta:
//   - Purpose: Bounded goroutine pool for batch-level parallelism; sized to GOMAXPROCS.
//   - Lifecycle: Operational after NewWorkerPool; drains and terminates on Stop.
//   - Concurrency: Safe; Submit blocks on backpressure rather than dropping jobs.
//   - Related: [NewWorkerPool], [Submit], [Stop].
type WorkerPool struct {
	jobs    chan func()
	wg      sync.WaitGroup
	once    sync.Once
	Workers int
}

// NewWorkerPool creates a pool sized to GOMAXPROCS so NN parallelism stays
// bounded by the same dial users already tune at the OS level.
//
// AI-Meta:
//   - Purpose: Construct and start a worker pool; goroutines run until Stop is called.
//   - Usage: p := network.NewWorkerPool(); defer p.Stop(); p.Submit(fn).
//   - Related: [WorkerPool], [Submit], [Stop].
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

// Submit enqueues fn for the next available worker. Blocks when all
// workers are busy — provides natural back-pressure to the caller.
//
// AI-Meta:
//   - Purpose: Dispatch a job to the pool; must not be called after Stop.
//   - Concurrency: Safe.
//   - Related: [Stop], [WorkerPool].
func (p *WorkerPool) Submit(fn func()) {
	p.jobs <- fn
}

// Stop closes the jobs channel and waits for all workers to drain and exit.
// Idempotent — multiple Stop calls are safe via sync.Once.
//
// AI-Meta:
//   - Purpose: Gracefully shut down the pool; call as defer p.Stop() after NewWorkerPool.
//   - Concurrency: Safe; blocks until the last in-flight job completes.
//   - Related: [NewWorkerPool], [Submit].
func (p *WorkerPool) Stop() {
	p.once.Do(func() {
		close(p.jobs)
	})
	p.wg.Wait()
}
