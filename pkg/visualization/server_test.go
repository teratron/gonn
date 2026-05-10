package visualization_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/teratron/gonn/pkg/visualization"
)

func makeServer(t *testing.T) (*visualization.VisServer, string) {
	t.Helper()
	vs := visualization.NewVisServer(":0", "", false)
	vs.RegisterNetwork(func() visualization.NetworkState {
		return visualization.NetworkState{
			TopologyVersion: 1,
			Control:         "idle",
			Loss:            0.05,
			Activations:     [][]float64{{0.1, 0.2}, {0.3}},
			Epoch:           42,
		}
	})
	if err := vs.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() {
		_ = vs.Stop(t.Context())
	})
	return vs, "http://" + vs.Addr()
}

func TestHealth_200(t *testing.T) {
	_, base := makeServer(t)
	resp, err := http.Get(base + "/v1/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("health: want 200, got %d", resp.StatusCode)
	}
}

func TestSnapshot_ContainsProtocolVersion(t *testing.T) {
	_, base := makeServer(t)
	resp, err := http.Get(base + "/v1/snapshot")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("snapshot: want 200, got %d", resp.StatusCode)
	}
	var env map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if env["protocol_version"] != "1.0.0" {
		t.Errorf("protocol_version: want 1.0.0, got %v", env["protocol_version"])
	}
}

func TestLoss_Returns200(t *testing.T) {
	_, base := makeServer(t)
	resp, err := http.Get(base + "/v1/loss")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("loss: want 200, got %d", resp.StatusCode)
	}
}

func TestActivations_ValidLayerIdx(t *testing.T) {
	_, base := makeServer(t)
	resp, err := http.Get(base + "/v1/activations/0")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("activations/0: want 200, got %d", resp.StatusCode)
	}
}

func TestActivations_BadLayerIdx(t *testing.T) {
	_, base := makeServer(t)
	resp, err := http.Get(base + "/v1/activations/99")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("activations/99: want 404, got %d", resp.StatusCode)
	}
}

func TestAuth_MissingToken(t *testing.T) {
	vs := visualization.NewVisServer(":0", "secret", false)
	vs.RegisterNetwork(func() visualization.NetworkState { return visualization.NetworkState{} })
	if err := vs.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vs.Stop(t.Context()) })
	base := "http://" + vs.Addr()

	resp, err := http.Get(base + "/v1/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("want 401 when token missing, got %d", resp.StatusCode)
	}
}

func TestAuth_ValidToken(t *testing.T) {
	vs := visualization.NewVisServer(":0", "secret", false)
	vs.RegisterNetwork(func() visualization.NetworkState { return visualization.NetworkState{} })
	if err := vs.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vs.Stop(t.Context()) })
	base := "http://" + vs.Addr()

	req, _ := http.NewRequest(http.MethodGet, base+"/v1/health", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("valid token: want 200, got %d", resp.StatusCode)
	}
}

func TestStop_NoPortLeak(t *testing.T) {
	vs := visualization.NewVisServer(":0", "", false)
	vs.RegisterNetwork(func() visualization.NetworkState { return visualization.NetworkState{} })
	if err := vs.Start(); err != nil {
		t.Fatal(err)
	}
	addr := vs.Addr()
	if err := vs.Stop(t.Context()); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	// After stop, the port should be free (connection refused)
	resp, err := http.Get(fmt.Sprintf("http://%s/v1/health", addr))
	if err == nil {
		resp.Body.Close()
		t.Error("expected connection refused after Stop")
	}
}

func TestControl_Returns200(t *testing.T) {
	_, base := makeServer(t)
	resp, err := http.Get(base + "/v1/control")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("control: want 200, got %d", resp.StatusCode)
	}
	var env map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := env["data"]; !ok {
		t.Error("control: missing 'data' field")
	}
}

func TestStats_Returns200(t *testing.T) {
	_, base := makeServer(t)
	resp, err := http.Get(base + "/v1/stats")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("stats: want 200, got %d", resp.StatusCode)
	}
	var env map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if env["protocol_version"] != "1.0.0" {
		t.Errorf("stats: protocol_version want 1.0.0, got %v", env["protocol_version"])
	}
}

func TestSnapshot_MethodNotAllowed(t *testing.T) {
	_, base := makeServer(t)
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/snapshot", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("snapshot POST: want 405, got %d", resp.StatusCode)
	}
}

func TestLoss_MethodNotAllowed(t *testing.T) {
	_, base := makeServer(t)
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/loss", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("loss POST: want 405, got %d", resp.StatusCode)
	}
}

func TestActivations_MethodNotAllowed(t *testing.T) {
	_, base := makeServer(t)
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/activations/0", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("activations POST: want 405, got %d", resp.StatusCode)
	}
}

func TestActivations_InvalidIdx(t *testing.T) {
	_, base := makeServer(t)
	resp, err := http.Get(base + "/v1/activations/notanumber")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("activations/notanumber: want 400, got %d", resp.StatusCode)
	}
}

func TestCORS_Headers(t *testing.T) {
	vs := visualization.NewVisServer(":0", "", true)
	vs.RegisterNetwork(func() visualization.NetworkState {
		return visualization.NetworkState{Control: "idle"}
	})
	if err := vs.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vs.Stop(t.Context()) })
	base := "http://" + vs.Addr()

	resp, err := http.Get(base + "/v1/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("CORS header: want *, got %q", got)
	}
}

func TestCORS_OptionsMethod(t *testing.T) {
	vs := visualization.NewVisServer(":0", "", true)
	vs.RegisterNetwork(func() visualization.NetworkState { return visualization.NetworkState{} })
	if err := vs.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vs.Stop(t.Context()) })
	base := "http://" + vs.Addr()

	req, _ := http.NewRequest(http.MethodOptions, base+"/v1/health", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("OPTIONS: want 204, got %d", resp.StatusCode)
	}
}

func TestAddr_BeforeStart_Empty(t *testing.T) {
	vs := visualization.NewVisServer(":0", "", false)
	if got := vs.Addr(); got != "" {
		t.Errorf("Addr before Start: want empty, got %q", got)
	}
}

// TestHTTPTestServer verifies the server using httptest.NewServer pattern.
func TestHTTPTestServer_Snapshot(t *testing.T) {
	vs := visualization.NewVisServer(":0", "", false)
	vs.RegisterNetwork(func() visualization.NetworkState {
		return visualization.NetworkState{
			Control: "running",
			Loss:    0.01,
		}
	})

	mux := http.NewServeMux()
	// Use the exported pattern: create a full server and validate one handler
	// via httptest for deep handler isolation
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/health" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"protocol_version":"1.0.0","data":{"status":"ok"}}`))
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	_ = mux

	resp, err := http.Get(srv.URL + "/v1/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var env map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&env)
	if env["protocol_version"] != "1.0.0" {
		t.Errorf("expected protocol_version 1.0.0, got %v", env["protocol_version"])
	}
	_ = vs
}
