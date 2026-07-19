// Package nn — visualization snapshot publishing.
//
// The Fit loop publishes an epoch-end NetworkState into an atomic pointer;
// the visualization server's SnapFn reads it lock-free on every HTTP
// request. Before the first epoch completes (or when the network is idle)
// the SnapFn falls back to a topology-only snapshot so /v1/snapshot is
// never empty (audit B7: the endpoint previously served a hardcoded stub).
package nn

import (
	"github.com/teratron/gonn/pkg/visualization"
)

// controlString maps the atomic control state to its JSON label.
func controlString(state int32) string {
	switch state {
	case controlRunning:
		return "running"
	case controlPaused:
		return "paused"
	case controlStopped:
		return "stopped"
	default:
		return "idle"
	}
}

// visLayerInfo assembles the LayerInfo slice describing the dense topology.
// Conv-prefix layers are intentionally out of scope for protocol 1.0.0 —
// the snapshot describes the trainable dense head.
func (n *NN[T]) visLayerInfo() []visualization.LayerInfo {
	layers := make([]visualization.LayerInfo, 0, len(n.Hiddens)+2)
	layers = append(layers, visualization.LayerInfo{Type: "input", Index: 0, Size: n.Network.Input.Len()})
	for i := range n.Hiddens {
		layers = append(layers, visualization.LayerInfo{
			Type: "hidden", Index: i + 1, Size: n.Hiddens[i].Len(),
		})
	}
	layers = append(layers, visualization.LayerInfo{
		Type: "output", Index: len(n.Hiddens) + 1, Size: n.Network.Output.Len(),
	})
	return layers
}

// buildVisState captures the network's observable state into a fresh
// NetworkState. Must be called while holding the training lock (Fit owns
// it for the whole run) so cell reads do not race weight updates.
func (n *NN[T]) buildVisState(lossVal float64, epoch, iteration uint64) *visualization.NetworkState {
	acts := make([][]float64, len(n.Hiddens))
	for i := range n.Hiddens {
		cells := n.Hiddens[i].Cells()
		row := make([]float64, len(cells))
		for j, h := range cells {
			row[j] = float64(*h.GetValue())
		}
		acts[i] = row
	}
	return &visualization.NetworkState{
		Layers:          n.visLayerInfo(),
		Control:         controlString(n.control.Load()),
		Activations:     acts,
		TopologyVersion: n.Network.TopologyVersion(),
		Loss:            lossVal,
		Epoch:           epoch,
		Iteration:       iteration,
	}
}

// publishVisSnapshot stores a fresh snapshot for the visualization server.
// No-op when WithVisualizationEndpoint was not configured — the epoch loop
// pays nothing for the feature it does not use.
func (n *NN[T]) publishVisSnapshot(lossVal float64, epoch, iteration uint64) {
	if n.vis == nil {
		return
	}
	n.visState.Store(n.buildVisState(lossVal, epoch, iteration))
}

// visSnapshot is the SnapFn registered with the visualization server. It
// prefers the last published training snapshot; before any epoch has
// completed it degrades to a topology-only view with live control state.
func (n *NN[T]) visSnapshot() visualization.NetworkState {
	if s := n.visState.Load(); s != nil {
		// Copy the struct so handlers never share the stored pointer's
		// slices with a concurrent Store (slices themselves are never
		// mutated after publish, so a shallow copy is safe).
		snap := *s
		snap.Control = controlString(n.control.Load())
		return snap
	}
	return visualization.NetworkState{
		Layers:          n.visLayerInfo(),
		Control:         controlString(n.control.Load()),
		TopologyVersion: n.Network.TopologyVersion(),
	}
}
