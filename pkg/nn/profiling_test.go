// Package nn — profiling (pprof) hook tests.
//
// The real listener is replaced via setListenAndServe so tests don't bind
// a port. Verifies that startProfilingServer is dispatched once per
// address and that listener errors are logged rather than escalated.
package nn

import (
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestStartProfilingServerEmptyAddrIsNoop(t *testing.T) {
	calls := new(atomic.Int32)
	prev := setListenAndServe(func(addr string, h http.Handler) error {
		calls.Add(1)
		return nil
	})
	t.Cleanup(func() { setListenAndServe(prev) })

	startProfilingServer("")
	time.Sleep(20 * time.Millisecond)
	if calls.Load() != 0 {
		t.Errorf("empty addr triggered %d calls", calls.Load())
	}
}

func TestStartProfilingServerOnceAcrossDuplicates(t *testing.T) {
	calls := new(atomic.Int32)
	gate := make(chan struct{})
	var gateOnce sync.Once
	closeGate := func() { gateOnce.Do(func() { close(gate) }) }

	prev := setListenAndServe(func(addr string, h http.Handler) error {
		calls.Add(1)
		<-gate // hold the goroutine alive while the test runs
		return nil
	})
	t.Cleanup(func() {
		closeGate()
		setListenAndServe(prev)
	})

	addr := ":0-test-once"
	startProfilingServer(addr)
	startProfilingServer(addr)
	startProfilingServer(addr)
	time.Sleep(50 * time.Millisecond)
	if got := calls.Load(); got != 1 {
		t.Errorf("listener spawned %d times, want 1", got)
	}
}

func TestStartProfilingServerSurvivesListenerError(t *testing.T) {
	done := make(chan struct{})
	prev := setListenAndServe(func(addr string, h http.Handler) error {
		defer close(done)
		return errors.New("synthetic listener failure")
	})
	t.Cleanup(func() { setListenAndServe(prev) })

	startProfilingServer(":0-test-error")
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("listener stub never invoked")
	}
	// Allow the warn log to flush before the test returns; the cleanup
	// only swaps the hook back, never the goroutine.
	time.Sleep(20 * time.Millisecond)
}
