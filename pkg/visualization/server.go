// Package visualization — HTTP observability adapter for GoNN networks.
//
// VisServer exposes a read-only JSON API over HTTP so external dashboards and
// monitoring tools can observe a running network without touching the training
// loop. The server is started by compile() when WithVisualizationEndpoint is
// configured; callers stop it via NN.Close().
//
// Endpoint layout (protocol_version 1.0.0):
//
//	GET /v1/snapshot     — full network state snapshot
//	GET /v1/loss         — current loss value
//	GET /v1/activations/{layerIdx} — hidden layer activations
//	GET /v1/control      — training control state
//	GET /v1/stats        — training statistics
//	GET /v1/health       — always 200 OK
package visualization

import (
	"context"
	"net"
	"net/http"
	"sync"
	"time"
)

const protocolVersion = "1.0.0"

// NetworkState is the read-only snapshot of a network's observable state.
// All numeric fields are float64 regardless of the network's generic T so
// the JSON API stays schema-stable across precision variants.
//
// AI-Meta:
//   - Purpose: Serialisable snapshot of NN observable state for HTTP endpoint responses.
//   - Usage: Returned by the SnapFn callback registered via RegisterNetwork.
//   - Related: [VisServer], [RegisterNetwork].
//   - Stability: Stable.
type NetworkState struct {
	Layers          []LayerInfo `json:"layers"`
	Control         string      `json:"control"`
	Activations     [][]float64 `json:"activations"`
	TopologyVersion uint64      `json:"topology_version"`
	Loss            float64     `json:"loss"`
	Epoch           uint64      `json:"epoch"`
	Iteration       uint64      `json:"iteration"`
}

// LayerInfo describes a single layer in the topology snapshot.
//
// AI-Meta:
//   - Purpose: Per-layer descriptor inside NetworkState.
//   - Related: [NetworkState].
//   - Stability: Stable.
type LayerInfo struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Size  int    `json:"size"`
}

// SnapFn is the callback type supplied to RegisterNetwork. The VisServer calls
// it on every incoming request to obtain a fresh snapshot without holding locks
// on the network itself.
//
// AI-Meta:
//   - Purpose: Callback type for pulling a fresh NetworkState snapshot from the NN.
//   - Related: [VisServer.RegisterNetwork], [NetworkState].
//   - Stability: Stable.
type SnapFn func() NetworkState

// VisServer is the HTTP observability adapter. A single server instance is
// created per compiled NN when WithVisualizationEndpoint is configured.
//
// AI-Meta:
//   - Purpose: HTTP server exposing network state for external dashboard tools.
//   - Lifecycle: Created by NewVisServer; started by Start; stopped by Stop.
//   - Concurrency: Start/Stop are safe for concurrent calls; handler callbacks must be goroutine-safe.
//   - Related: [NewVisServer], [RegisterNetwork], [Start], [Stop].
//   - Stability: Stable.
type VisServer struct {
	listener net.Listener
	srv      *http.Server
	snapFn   SnapFn
	addr     string
	token    string
	mu       sync.RWMutex
	cors     bool
}

// NewVisServer creates a VisServer that will listen on addr. token enables
// bearer-token authentication; cors enables CORS headers.
//
// AI-Meta:
//   - Purpose: Construct a VisServer with address, auth, and CORS settings.
//   - Usage: vs := visualization.NewVisServer(":8080", "", false).
//   - Related: [VisServer], [RegisterNetwork], [Start].
//   - Stability: Stable.
func NewVisServer(addr, token string, cors bool) *VisServer {
	return &VisServer{addr: addr, token: token, cors: cors}
}

// RegisterNetwork sets the callback used by all handlers to pull current state.
// Must be called before Start. Replaces any previously registered callback.
//
// AI-Meta:
//   - Purpose: Register the NN-supplied snapshot callback before starting the server.
//   - Related: [VisServer], [SnapFn].
//   - Stability: Stable.
func (vs *VisServer) RegisterNetwork(fn SnapFn) {
	vs.mu.Lock()
	vs.snapFn = fn
	vs.mu.Unlock()
}

// snapOrEmpty returns the current snapshot or a zero-value fallback when no
// callback is registered.
func (vs *VisServer) snapOrEmpty() NetworkState {
	vs.mu.RLock()
	fn := vs.snapFn
	vs.mu.RUnlock()
	if fn == nil {
		return NetworkState{Control: "idle"}
	}
	return fn()
}

// Start builds the mux, binds the listener, and launches the HTTP server in a
// background goroutine. Returns the error from net.Listen if binding fails.
// The caller can retrieve the actual bound address via Addr() for :0 listeners.
//
// AI-Meta:
//   - Purpose: Bind the listener and start the HTTP server; non-blocking.
//   - Errors: Returns net.Listen error on bind failure.
//   - Concurrency: Safe; subsequent calls are no-ops if already started.
//   - Related: [Stop], [Addr].
//   - Stability: Stable.
func (vs *VisServer) Start() error {
	mux := http.NewServeMux()
	vs.registerRoutes(mux)
	ln, err := net.Listen("tcp", vs.addr)
	if err != nil {
		return err
	}
	vs.listener = ln
	vs.srv = &http.Server{
		Handler:        authMiddleware(vs.token, corsMiddleware(vs.cors, mux)),
		ReadTimeout:    5 * time.Second,
		WriteTimeout:   10 * time.Second,
		IdleTimeout:    60 * time.Second,
		MaxHeaderBytes: 1 << 16, // 64 KiB — the API takes no large headers
	}
	go func() { _ = vs.srv.Serve(ln) }()
	return nil
}

// Stop gracefully shuts down the HTTP server using ctx for timeout control.
// Passing context.Background() with a short deadline is recommended.
//
// AI-Meta:
//   - Purpose: Gracefully stop the HTTP server; used by NN.Close.
//   - Errors: Returns error from http.Server.Shutdown.
//   - Related: [Start], [NN.Close].
//   - Stability: Stable.
func (vs *VisServer) Stop(ctx context.Context) error {
	if vs.srv == nil {
		return nil
	}
	return vs.srv.Shutdown(ctx)
}

// Addr returns the actual bound address after Start. Returns empty string
// before Start is called.
//
// AI-Meta:
//   - Purpose: Return the actual bound address for :0 listeners used in tests.
//   - Related: [Start].
//   - Stability: Stable.
func (vs *VisServer) Addr() string {
	if vs.listener == nil {
		return ""
	}
	return vs.listener.Addr().String()
}

// registerRoutes wires the six observability endpoints onto mux.
func (vs *VisServer) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/snapshot", vs.snapshotHandler)
	mux.HandleFunc("/v1/loss", vs.lossHandler)
	mux.HandleFunc("/v1/activations/", vs.activationsHandler)
	mux.HandleFunc("/v1/control", vs.controlHandler)
	mux.HandleFunc("/v1/stats", vs.statsHandler)
	mux.HandleFunc("/v1/health", vs.healthHandler)
}
