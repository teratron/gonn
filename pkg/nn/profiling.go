// Package nn — pprof opt-in hook.
//
// Implements [l2-perf-impl] §5.4 (PERF-5). WithProfiling stores the listen
// address; startProfilingServer is called from compile() at the end of a
// successful build. The handlers are registered on a dedicated mux — NOT
// http.DefaultServeMux — so enabling GoNN profiling never exposes whatever
// else the host process may have registered globally (audit E). Server
// handles are retained so NN.Close can shut the listener down gracefully.
package nn

import (
	"context"
	"net"
	"net/http"
	"net/http/pprof"
	"sync"
	"time"

	"github.com/teratron/gonn/pkg/utils"
)

// profilingMu guards the servers map and the listen hook so tests can swap
// in a stub without racing the worker goroutines. One server is kept per
// address across the process — repeated New / Compile calls with the same
// addr never spawn duplicate goroutines or hit "address in use" errors.
var (
	profilingMu      sync.Mutex
	profilingServers = make(map[string]*http.Server)
	listenAndServeFn = defaultListenAndServe
)

// isLoopbackAddr reports whether addr binds only the loopback interface. Blank
// host (":6060") binds all interfaces and is treated as non-loopback.
func isLoopbackAddr(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}
	if host == "" {
		return false
	}
	if host == "localhost" {
		return true
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	return false
}

// pprofMux builds a fresh mux carrying only the /debug/pprof handlers.
func pprofMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	return mux
}

// defaultListenAndServe is production's bind. Tests swap a stub via
// setListenAndServe.
func defaultListenAndServe(srv *http.Server) error {
	return srv.ListenAndServe()
}

// setListenAndServe replaces the bind hook under the same mutex that
// guards reads. Returns the previous hook so tests can restore it.
func setListenAndServe(fn func(*http.Server) error) func(*http.Server) error {
	profilingMu.Lock()
	defer profilingMu.Unlock()
	prev := listenAndServeFn
	listenAndServeFn = fn
	return prev
}

// getListenAndServe loads the current hook under the mutex so the
// goroutine's read happens-before any concurrent setListenAndServe.
func getListenAndServe() func(*http.Server) error {
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
	// pprof exposes heap/goroutine profiles that can contain training data.
	// Warn loudly when it is bound to a non-loopback address so an operator
	// does not unknowingly expose it to the network (audit E).
	if !isLoopbackAddr(addr) {
		utils.Logger.Warn("pprof profiling endpoint bound to a non-loopback address — "+
			"exposes /debug/pprof to the network; prefer 127.0.0.1", "addr", addr)
	}
	profilingMu.Lock()
	if _, running := profilingServers[addr]; running {
		profilingMu.Unlock()
		return
	}
	srv := &http.Server{
		Addr:              addr,
		Handler:           pprofMux(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	profilingServers[addr] = srv
	profilingMu.Unlock()

	go func() {
		fn := getListenAndServe()
		if err := fn(srv); err != nil && err != http.ErrServerClosed {
			utils.Logger.Warn("pprof listener exited", "addr", addr, "err", err.Error())
		}
	}()
}

// stopProfilingServer gracefully shuts down the pprof listener bound to addr
// (no-op for "" or an address never started). The map entry is removed so a
// later Compile with the same addr can restart it. Called by NN.Close; note
// that NN instances sharing one profiling address share one listener, so the
// first Close wins — pprof is process-level observability, not per-network.
func stopProfilingServer(ctx context.Context, addr string) error {
	if addr == "" {
		return nil
	}
	profilingMu.Lock()
	srv, ok := profilingServers[addr]
	if ok {
		delete(profilingServers, addr)
	}
	profilingMu.Unlock()
	if !ok {
		return nil
	}
	return srv.Shutdown(ctx)
}
