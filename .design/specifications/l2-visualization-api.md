# Visualization API

**Version:** 0.1.0
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

## 6. Implementation Notes

1. New file `pkg/nn/visualization.go` (or `pkg/visualization/server.go`).
2. Server lifecycle bound to `nn.NN[T]` — stops when network is garbage-collected or `Close()` called.
3. CORS headers permissive only when explicitly enabled (`WithVisualizationCORS(true)`).

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[VIS-DIR]` | `pkg/nn/` (or new `pkg/visualization/`) | Implementation home |
| `[GUI-REPO]` | (external — TBD URL) | Reference visualizer consuming this API |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #7. |
| 0.1.0 | 2026-05-07 | [Pre-Plan] Trust Mode promoted Draft → Stable. MVC satisfied (Overview + Invariant Compliance + Canonical References). |
