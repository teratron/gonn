// Package network — benchmark suite (PERF-1).
//
// Naming follows [l2-perf-impl] §5: Benchmark<Op>_<Topology>_<DType>.
// Run via:
//
//	go test -bench=. -benchmem -count=5 ./pkg/network/
package network

import "testing"

// BenchmarkAcquireRelease_F32 measures the buffer-pool round-trip
// hot path. PERF-4 expects zero allocations per iteration once the
// pool is warm.
func BenchmarkAcquireRelease_F32(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		buf := AcquireActivations[float32](64)
		ReleaseActivations(buf)
	}
}

// BenchmarkAcquireRelease_F64 mirrors the F32 case for the float64
// pool to guard parity between the two specialisations.
func BenchmarkAcquireRelease_F64(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		buf := AcquireActivations[float64](64)
		ReleaseActivations(buf)
	}
}

// BenchmarkPreallocStorage_DeepNetwork_F32 stresses the Compile-time
// preallocation hook with a synthetic deep-network shape so PERF-2
// regressions surface even before training kicks in.
func BenchmarkPreallocStorage_DeepNetwork_F32(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = NewPreallocStorage[float32](16_384, 256, 64)
	}
}

// BenchmarkWorkerPool_Submit measures the cost of fan-out through the
// worker pool. Per PERF-3 the pool size is bounded by GOMAXPROCS.
func BenchmarkWorkerPool_Submit(b *testing.B) {
	pool := NewWorkerPool()
	defer pool.Stop()
	b.ReportAllocs()
	for b.Loop() {
		done := make(chan struct{})
		pool.Submit(func() { close(done) })
		<-done
	}
}
