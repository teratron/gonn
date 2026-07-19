# Visualization API

**Version:** 0.2.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-observability-protocol.md

## Overview

HTTP/JSON adapter exposing `l1-observability-protocol.md` to external visualizer repositories
(planned separate repo with web UI / desktop GUI). Defines the wire format, endpoints, and
authentication surface so that GUIs developed independently can connect to a running GoNN training
session.

## Related Specifications

- [l1-observability-protocol.md](l1-observability-protocol.md) — Parent — read-only state contract
- [l1-training-control.md](l1-training-control.md) — Control commands sent back from GUI (out of scope here — see §5.3)

## 1. Motivation

The visualizer GUI is intentionally kept in a separate repository to:

- Avoid dragging UI dependencies (templates, JS, frontend bundlers) into the core library.
- Allow community alternative visualizers (terminal UI, Jupyter widget, web page).
- Enable the library to be used **without** any UI weight — `import "github.com/teratron/gonn"`
  remains zero-dep.

To make that boundary concrete, the library publishes a documented, versioned wire protocol.

## 2. Constraints & Assumptions

- Stdlib `net/http` only (per `C29`). No external HTTP frameworks.
- JSON over HTTP/1.1 — the simplest cross-language transport.
- The visualization server is **opt-in** — `nn.WithVisualizationEndpoint(addr string)` starts an
  HTTP listener on the given address. Default: disabled.
- Authentication: optional bearer-token via `WithVisualizationToken(string)`; default = no auth, only
  bind to localhost.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| OBS-1 (Snapshots) | HTTP responses serialize a `Snapshot()` copy; no live references leave the process. |
| OBS-2 (Bounded read cost) | Endpoints map 1:1 to L1 read surfaces with the same complexity. |
| OBS-3 (Zero overhead when off) | If `WithVisualizationEndpoint` not called, no HTTP server is started; no goroutines spawned. |
| OBS-4 (Versioned) | Every response includes `"protocol_version": "X.Y.Z"`; breaking changes bump major. |

## 5. Detailed Design

### 5.1 HTTP Endpoints (proposed)

| Method | Path | Returns |
| :--- | :--- | :--- |
| GET | `/v1/snapshot` | Full snapshot JSON |
| GET | `/v1/loss?n=100` | Last N loss values |
| GET | `/v1/activations/{layerIdx}` | Layer activation vector |
| GET | `/v1/control` | Current `ControlState()` |
| GET | `/v1/stats` | Counters |
| GET | `/v1/health` | Liveness for visualizers |

### 5.2 Wire Format Example

```json
{
  "protocol_version": "1.0.0",
  "iteration": 4250,
  "loss": 0.0123,
  "control_state": "Running",
  "elapsed_seconds": 17.4
}
```

### 5.3 Out of Scope (for v1)

- Sending **control commands** (Pause/Resume) from the GUI back to the library — reserved for a future
  `l2-control-api.md` companion spec. v1 is read-only.
- WebSocket streaming for high-frequency updates. v1 uses polling; clients pace themselves.
- Authentication beyond a bearer token; OIDC/mTLS deferred.

## 5.4 Package Structure

```plaintext
pkg/visualization/
├── server.go        # VisServer struct, Start()/Stop(), http.ServeMux registration
├── handlers.go      # handler functions: snapshotHandler, lossHandler, activationsHandler, etc.
├── middleware.go    # bearer-token auth middleware (no-op when token not set)
└── server_test.go   # httptest.NewServer-based integration tests
```

`pkg/nn/` option wiring:

```plaintext
pkg/nn/
├── options.go   # WithVisualizationEndpoint[T](addr string), WithVisualizationToken[T](token string), WithVisualizationCORS[T](enabled bool)
└── compile.go   # starts pkg/visualization.VisServer after compile() if endpoint configured
```

### 5.5 Server Struct and Lifecycle

```go
// [REFERENCE]
type VisServer struct {
    addr    string
    token   string
    cors    bool
    mux     *http.ServeMux
    srv     *http.Server
    cancel  context.CancelFunc
}

func NewVisServer(addr, token string, cors bool) *VisServer
func (s *VisServer) RegisterNetwork(snap func() Snapshot) // called by compile()
func (s *VisServer) Start() error                          // non-blocking; spawns goroutine
func (s *VisServer) Stop(ctx context.Context) error        // graceful shutdown
```

`NN[T].Close()` must call `s.Stop(ctx)` if a server was registered — lifecycle bound to NN.

### 5.6 Handler Signatures

```go
// [REFERENCE]
func (s *VisServer) snapshotHandler(w http.ResponseWriter, r *http.Request)       // GET /v1/snapshot
func (s *VisServer) lossHandler(w http.ResponseWriter, r *http.Request)           // GET /v1/loss?n=100
func (s *VisServer) activationsHandler(w http.ResponseWriter, r *http.Request)    // GET /v1/activations/{layerIdx}
func (s *VisServer) controlHandler(w http.ResponseWriter, r *http.Request)        // GET /v1/control
func (s *VisServer) statsHandler(w http.ResponseWriter, r *http.Request)          // GET /v1/stats
func (s *VisServer) healthHandler(w http.ResponseWriter, r *http.Request)         // GET /v1/health
```

All handlers write `Content-Type: application/json` and include `"protocol_version": "1.0.0"`.
Non-existing `layerIdx` returns 404 with `{"error":"layer not found"}`.

## 6. Implementation Notes

1. Implement `pkg/visualization/server.go` + `handlers.go` — no `pkg/nn` changes yet.
2. Write `server_test.go` using `httptest.NewServer` for endpoint coverage.
3. Wire `WithVisualizationEndpoint` option in `pkg/nn/options.go` and `compile.go`.
4. Add `Close()` to `pkg/nn/nn.go` if not present; call `VisServer.Stop()` there.
5. CORS headers permissive only when explicitly enabled (`WithVisualizationCORS(true)`).

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[VIS-SERVER]` | `pkg/visualization/server.go` | VisServer struct and lifecycle |
| `[VIS-HANDLERS]` | `pkg/visualization/handlers.go` | Endpoint handler implementations |
| `[NN-OPT]` | `pkg/nn/options.go` | WithVisualizationEndpoint option wiring |
| `[GUI-REPO]` | (external — TBD URL) | Reference visualizer consuming this API |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #7. |
| 0.1.0 | 2026-05-07 | [Pre-Plan] Trust Mode promoted Draft → Stable. MVC satisfied. |
| 0.2.0 | 2026-05-10 | Added §5.4 package structure, §5.5 VisServer lifecycle, §5.6 handler signatures, updated canonical references. |
