// Package nn — pprof opt-in hook.
//
// Implements [l2-perf-impl] §5.4 (PERF-5). Importing net/http/pprof for
// its side effects registers /debug/pprof/* handlers on
// http.DefaultServeMux. WithProfiling stores the listen address;
// startProfilingServer is called from compile() at the end of a
// successful build.
package nn

import (
	"net/http"
	_ "net/http/pprof" // pprof handler registration side effect
	"sync"
	"time"

	"github.com/teratron/gonn/pkg/utils"
)

// profilingOnce guarantees a single listener per address across the
// process — repeated New / Compile calls with the same addr never spawn
// duplicate goroutines or hit "address in use" errors. The same mutex
// also protects the listenAndServe function pointer so tests can swap
// in a stub without racing the worker goroutines.
var (
	profilingMu      sync.Mutex
	profilingStarted = make(map[string]bool)
	listenAndServeFn = defaultListenAndServe
)

// defaultListenAndServe is production's bind. Tests swap a stub via
// setListenAndServe.
func defaultListenAndServe(addr string, handler http.Handler) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
	}
	return srv.ListenAndServe()
}

// setListenAndServe replaces the bind hook under the same mutex that
// guards reads. Returns the previous hook so tests can restore it.
func setListenAndServe(fn func(string, http.Handler) error) func(string, http.Handler) error {
	profilingMu.Lock()
	defer profilingMu.Unlock()
	prev := listenAndServeFn
	listenAndServeFn = fn
	return prev
}

// getListenAndServe loads the current hook under the mutex so the
// goroutine's read happens-before any concurrent setListenAndServe.
func getListenAndServe() func(string, http.Handler) error {
	profilingMu.Lock()
	defer profilingMu.Unlock()
	return listenAndServeFn
}

// startProfilingServer launches the pprof listener on addr if it has
// not already been started in this process. Errors from
// ListenAndServe are logged at Warn level — the listener crashing is
// non-fatal because pprof is observability-only.
func startProfilingServer(addr string) {
	if addr == "" {
		return
	}
	profilingMu.Lock()
	if profilingStarted[addr] {
		profilingMu.Unlock()
		return
	}
	profilingStarted[addr] = true
	profilingMu.Unlock()

	go func() {
		fn := getListenAndServe()
		if err := fn(addr, nil); err != nil {
			utils.Logger.Warn("pprof listener exited", "addr", addr, "err", err.Error())
		}
	}()
}
