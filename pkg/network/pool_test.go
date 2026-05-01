// Package network — pool / worker correctness tests.
package network

import (
	"sync/atomic"
	"testing"
)

func TestAcquireActivationsF32(t *testing.T) {
	buf := AcquireActivations[float32](16)
	if len(buf) != 16 {
		t.Errorf("len = %d, want 16", len(buf))
	}
	for i, v := range buf {
		if v != 0 {
			t.Errorf("buf[%d] = %v, want 0", i, v)
		}
	}
	buf[3] = 7
	ReleaseActivations(buf)

	// Acquiring a smaller size from the pool must yield zeroed slice.
	again := AcquireActivations[float32](4)
	for _, v := range again {
		if v != 0 {
			t.Errorf("released buffer not zeroed: %v", v)
		}
	}
	ReleaseActivations(again)
}

func TestAcquireActivationsF64(t *testing.T) {
	buf := AcquireActivations[float64](32)
	if len(buf) != 32 {
		t.Errorf("len = %d, want 32", len(buf))
	}
	ReleaseActivations(buf)
}

func TestAcquireZeroAndNegative(t *testing.T) {
	if got := AcquireActivations[float32](0); len(got) != 0 {
		t.Errorf("size 0 → len %d", len(got))
	}
	if got := AcquireActivations[float32](-5); len(got) != 0 {
		t.Errorf("negative size → len %d", len(got))
	}
	ReleaseActivations[float32](nil) // no-op
}

func TestAcquireGrowsBeyondPoolCapacity(t *testing.T) {
	huge := AcquireActivations[float32](4096)
	if len(huge) != 4096 {
		t.Errorf("grow failed: len %d", len(huge))
	}
	ReleaseActivations(huge)
}

func TestPreallocStorageSizing(t *testing.T) {
	s := NewPreallocStorage[float64](100, 16, 8)
	if len(s.WeightStorage) != 100 || len(s.ActivationStorage) != 16 ||
		len(s.GradientStorage) != 16 || len(s.BiasStorage) != 8 {
		t.Errorf("storage sizes wrong: %+v", s)
	}
}

func TestWorkerPoolDispatchesAllJobs(t *testing.T) {
	pool := NewWorkerPool()
	defer pool.Stop()
	var counter atomic.Int32
	const N = 100
	done := make(chan struct{}, N)
	for range N {
		pool.Submit(func() {
			counter.Add(1)
			done <- struct{}{}
		})
	}
	for range N {
		<-done
	}
	if counter.Load() != N {
		t.Errorf("counter = %d, want %d", counter.Load(), N)
	}
}

func TestWorkerPoolStopIdempotent(t *testing.T) {
	pool := NewWorkerPool()
	pool.Stop()
	pool.Stop() // must not panic
}

func TestWorkerPoolWorkersBounded(t *testing.T) {
	pool := NewWorkerPool()
	defer pool.Stop()
	if pool.Workers < 1 {
		t.Errorf("Workers = %d, want >= 1", pool.Workers)
	}
}
