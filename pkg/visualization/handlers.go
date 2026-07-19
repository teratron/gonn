package visualization

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

// envelope wraps every response with protocol_version so clients can detect
// schema incompatibilities without inspecting headers.
type envelope struct {
	Data            any    `json:"data"`
	ProtocolVersion string `json:"protocol_version"`
}

func writeJSON(w http.ResponseWriter, code int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(envelope{ProtocolVersion: protocolVersion, Data: data})
}

func (vs *VisServer) snapshotHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, vs.snapOrEmpty())
}

func (vs *VisServer) lossHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	state := vs.snapOrEmpty()
	writeJSON(w, http.StatusOK, map[string]float64{"loss": state.Loss})
}

func (vs *VisServer) activationsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	// path: /v1/activations/{layerIdx}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/v1/activations/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing layerIdx"})
		return
	}
	idx, err := strconv.Atoi(parts[0])
	if err != nil || idx < 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid layerIdx"})
		return
	}
	state := vs.snapOrEmpty()
	if idx >= len(state.Activations) {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "layerIdx out of range"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"layer_idx":   idx,
		"activations": state.Activations[idx],
	})
}

func (vs *VisServer) controlHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	state := vs.snapOrEmpty()
	writeJSON(w, http.StatusOK, map[string]string{"control": state.Control})
}

func (vs *VisServer) statsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	state := vs.snapOrEmpty()
	writeJSON(w, http.StatusOK, map[string]any{
		"epoch":            state.Epoch,
		"iteration":        state.Iteration,
		"topology_version": state.TopologyVersion,
	})
}

func (vs *VisServer) healthHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}
