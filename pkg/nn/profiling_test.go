// Package nn — profiling (pprof) hook tests.
//
// The real listener is replaced via setListenAndServe so tests don't bind
// a port. Verifies that startProfilingServer is dispatched once per
// address, that listener errors are logged rather than escalated, and
// that stopProfilingServer releases the address for reuse.
package nn

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestStartProfilingServerEmptyAddrIsNoop(t *testing.T) {
	calls := new(atomic.Int32)
	prev := setListenAndServe(func(srv *http.Server) error {
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

	prev := setListenAndServe(func(srv *http.Server) error {
		calls.Add(1)
		<-gate // hold the goroutine alive while the test runs
		return nil
	})
	addr := ":0-test-once"
	t.Cleanup(func() {
		closeGate()
		setListenAndServe(prev)
		_ = stopProfilingServer(context.Background(), addr)
	})

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
	prev := setListenAndServe(func(srv *http.Server) error {
		defer close(done)
		return errors.New("synthetic listener failure")
	})
	addr := ":0-test-error"
	t.Cleanup(func() {
		setListenAndServe(prev)
		_ = stopProfilingServer(context.Background(), addr)
	})

	startProfilingServer(addr)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("listener stub never invoked")
	}
	// Allow the warn log to flush before the test returns; the cleanup
	// only swaps the hook back, never the goroutine.
	time.Sleep(20 * time.Millisecond)
}

func TestStopProfilingServerAllowsRestart(t *testing.T) {
	calls := new(atomic.Int32)
	prev := setListenAndServe(func(srv *http.Server) error {
		calls.Add(1)
		return http.ErrServerClosed // simulate a Shutdown-terminated server
	})
	addr := ":0-test-restart"
	t.Cleanup(func() {
		setListenAndServe(prev)
		_ = stopProfilingServer(context.Background(), addr)
	})

	startProfilingServer(addr)
	if err := stopProfilingServer(context.Background(), addr); err != nil {
		t.Fatalf("stopProfilingServer: %v", err)
	}
	startProfilingServer(addr) // after Stop the address must be reusable
	time.Sleep(50 * time.Millisecond)
	if got := calls.Load(); got != 2 {
		t.Errorf("listener spawned %d times across stop/restart, want 2", got)
	}
}
