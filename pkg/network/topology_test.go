package network

import (
	"errors"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// newDynamicNet builds a 2-hidden network with TopologyMode = Dynamic.
func newDynamicNet[T utils.Float]() (*Network[T], error) {
	in := layer.NewInput[T](3)
	h1 := layer.NewDense[T](4, activation.SIGMOID, false)
	h2 := layer.NewDense[T](4, activation.SIGMOID, false)
	out := layer.NewOutput[T](2, activation.SIGMOID, loss.MSE, false)
	n := New[T]()
	if err := n.SetLayers(in, []*layer.Dense[T]{h1, h2}, out); err != nil {
		return nil, err
	}
	if err := n.Build(); err != nil {
		return nil, err
	}
	n.topologyMode = Dynamic
	return &n, nil
}

// ────────────────────────────────────────────────────────────
// AddNeuron / RemoveNeuron
// ────────────────────────────────────────────────────────────

func TestAddNeuron_GrowsLayer(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	before := n.Hiddens[0].Len()
	if err := n.AddNeuron(0, 2); err != nil {
		t.Fatalf("AddNeuron: %v", err)
	}
	after := n.Hiddens[0].Len()
	if after != before+2 {
		t.Errorf("layer 0 size: want %d, got %d", before+2, after)
	}
	if n.TopologyVersion() != 1 {
		t.Errorf("TopologyVersion: want 1, got %d", n.TopologyVersion())
	}
	// Verify axon counts on successor layer
	expectedAxonsPerCell := n.Hiddens[0].Len() // each H1 cell has one axon per H0 cell
	for _, h := range n.Hiddens[1].cells {
		if len(h.Axons) != expectedAxonsPerCell {
			t.Errorf("H1 cell axon count: want %d, got %d", expectedAxonsPerCell, len(h.Axons))
			break
		}
	}
}

func TestAddNeuron_ImmutableRejected(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	n.topologyMode = Immutable
	err = n.AddNeuron(0, 1)
	if !errors.Is(err, utils.ErrImmutableMode) {
		t.Errorf("expected ErrImmutableMode, got %v", err)
	}
}

func TestAddNeuron_InvalidLayerIdx(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	err = n.AddNeuron(99, 1)
	if !errors.Is(err, utils.ErrInvalidPosition) {
		t.Errorf("expected ErrInvalidPosition, got %v", err)
	}
}

func TestAddNeuron_ZeroCount(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	err = n.AddNeuron(0, 0)
	if !errors.Is(err, utils.ErrEmptyLayer) {
		t.Errorf("expected ErrEmptyLayer, got %v", err)
	}
}

func TestRemoveNeuron_ShrinksLayer(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	before := n.Hiddens[0].Len()
	if err := n.RemoveNeuron(0, 1); err != nil {
		t.Fatalf("RemoveNeuron: %v", err)
	}
	if n.Hiddens[0].Len() != before-1 {
		t.Errorf("layer 0 size: want %d, got %d", before-1, n.Hiddens[0].Len())
	}
}

func TestRemoveNeuron_LeavesEmptyLayer(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	// Removing all cells should return ErrEmptyLayer
	err = n.RemoveNeuron(0, uint(n.Hiddens[0].Len()))
	if !errors.Is(err, utils.ErrEmptyLayer) {
		t.Errorf("expected ErrEmptyLayer, got %v", err)
	}
}

func TestRemoveNeuron_RollbackOnError(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	beforeLen := n.Hiddens[0].Len()
	beforeVer := n.TopologyVersion()
	_ = n.RemoveNeuron(0, uint(n.Hiddens[0].Len())) // should fail and rollback
	if n.Hiddens[0].Len() != beforeLen {
		t.Errorf("rollback failed: layer size changed from %d to %d", beforeLen, n.Hiddens[0].Len())
	}
	if n.TopologyVersion() != beforeVer {
		t.Errorf("rollback: TopologyVersion should not change on error")
	}
}

// ────────────────────────────────────────────────────────────
// AddHiddenLayer / RemoveHiddenLayer
// ────────────────────────────────────────────────────────────

func TestAddHiddenLayer_MidChain(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	beforeLen := len(n.Hiddens)
	if err := n.AddHiddenLayer(1, 5, activation.ReLU, false); err != nil {
		t.Fatalf("AddHiddenLayer: %v", err)
	}
	if len(n.Hiddens) != beforeLen+1 {
		t.Errorf("hidden count: want %d, got %d", beforeLen+1, len(n.Hiddens))
	}
	if n.Hiddens[1].Len() != 5 {
		t.Errorf("new layer size: want 5, got %d", n.Hiddens[1].Len())
	}
	if n.TopologyVersion() != 1 {
		t.Errorf("TopologyVersion: want 1, got %d", n.TopologyVersion())
	}
	// Verify that Hiddens[2] (formerly Hiddens[1]) now receives axons from Hiddens[1]
	for _, h := range n.Hiddens[2].cells {
		if len(h.Axons) != n.Hiddens[1].Len() {
			t.Errorf("H2 axon count: want %d (from new layer), got %d", n.Hiddens[1].Len(), len(h.Axons))
			break
		}
	}
}

func TestAddHiddenLayer_ImmutableRejected(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	n.topologyMode = Immutable
	err = n.AddHiddenLayer(0, 4, activation.SIGMOID, false)
	if !errors.Is(err, utils.ErrImmutableMode) {
		t.Errorf("expected ErrImmutableMode, got %v", err)
	}
}

func TestAddHiddenLayer_InvalidPosition(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	err = n.AddHiddenLayer(99, 4, activation.SIGMOID, false)
	if !errors.Is(err, utils.ErrInvalidPosition) {
		t.Errorf("expected ErrInvalidPosition, got %v", err)
	}
}

func TestAddHiddenLayer_ZeroSize(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	err = n.AddHiddenLayer(0, 0, activation.SIGMOID, false)
	if !errors.Is(err, utils.ErrEmptyLayer) {
		t.Errorf("expected ErrEmptyLayer, got %v", err)
	}
}

func TestRemoveHiddenLayer_Removes(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	beforeLen := len(n.Hiddens)
	if err := n.RemoveHiddenLayer(0); err != nil {
		t.Fatalf("RemoveHiddenLayer: %v", err)
	}
	if len(n.Hiddens) != beforeLen-1 {
		t.Errorf("hidden count: want %d, got %d", beforeLen-1, len(n.Hiddens))
	}
}

func TestRemoveHiddenLayer_ErrMinimumTopology(t *testing.T) {
	in := layer.NewInput[float64](2)
	h := layer.NewDense[float64](3, activation.SIGMOID, false)
	out := layer.NewOutput[float64](1, activation.SIGMOID, loss.MSE, false)
	n := New[float64]()
	_ = n.SetLayers(in, []*layer.Dense[float64]{h}, out)
	_ = n.Build()
	n.topologyMode = Dynamic

	err := n.RemoveHiddenLayer(0)
	if !errors.Is(err, utils.ErrMinimumTopology) {
		t.Errorf("expected ErrMinimumTopology, got %v", err)
	}
}

func TestTopologyVersion_IncrementPerCommit(t *testing.T) {
	n, err := newDynamicNet[float64]()
	if err != nil {
		t.Fatal(err)
	}
	if v := n.TopologyVersion(); v != 0 {
		t.Errorf("initial TopologyVersion: want 0, got %d", v)
	}
	_ = n.AddNeuron(0, 1)
	if v := n.TopologyVersion(); v != 1 {
		t.Errorf("after AddNeuron: want 1, got %d", v)
	}
	_ = n.RemoveNeuron(1, 1)
	if v := n.TopologyVersion(); v != 2 {
		t.Errorf("after RemoveNeuron: want 2, got %d", v)
	}
}
