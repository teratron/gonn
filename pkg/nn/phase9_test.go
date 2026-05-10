package nn_test

import (
	"errors"
	"io"
	"log/slog"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// buildDynamic compiles a 2-hidden-layer Dynamic network.
func buildDynamic(t *testing.T) *nn.NN[float64] {
	t.Helper()
	n, err := nn.New[float64](
		nn.WithInput[float64](3),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](2, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithTopologyMode[float64](network.Dynamic),
	)
	if err != nil {
		t.Fatalf("buildDynamic: %v", err)
	}
	return n
}

// buildImmutable compiles a simple Immutable network (default mode).
func buildImmutable(t *testing.T) *nn.NN[float64] {
	t.Helper()
	n, err := nn.New[float64](
		nn.WithInput[float64](3),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](2, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
	)
	if err != nil {
		t.Fatalf("buildImmutable: %v", err)
	}
	return n
}

// ============================================================================
// WithTopologyMode option
// ============================================================================

func TestWithTopologyMode_Dynamic_AllowsMutation(t *testing.T) {
	t.Parallel()
	n := buildDynamic(t)
	if err := n.AddNeuron(0, 1); err != nil {
		t.Errorf("AddNeuron on Dynamic network: unexpected error: %v", err)
	}
}

func TestWithTopologyMode_Immutable_RejectsAddNeuron(t *testing.T) {
	t.Parallel()
	n := buildImmutable(t)
	err := n.AddNeuron(0, 1)
	if err == nil {
		t.Fatal("AddNeuron on Immutable network: want error, got nil")
	}
	if !errors.Is(err, utils.ErrImmutableMode) {
		t.Errorf("want ErrImmutableMode, got %v", err)
	}
}

// ============================================================================
// WithLogger option
// ============================================================================

func TestWithLogger_Accepted(t *testing.T) {
	t.Parallel()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	n, err := nn.New[float64](
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](3, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithLogger[float64](logger),
	)
	if err != nil {
		t.Fatalf("New with WithLogger: %v", err)
	}
	if n == nil {
		t.Error("expected non-nil NN")
	}
}

// ============================================================================
// Visualization options
// ============================================================================

func TestWithVisualizationEndpoint_StartsServer(t *testing.T) {
	t.Parallel()
	n, err := nn.New[float64](
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](3, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithVisualizationEndpoint[float64](":0"),
	)
	if err != nil {
		t.Fatalf("New with WithVisualizationEndpoint: %v", err)
	}
	if closeErr := n.Close(); closeErr != nil {
		t.Errorf("Close: %v", closeErr)
	}
}

func TestWithVisualizationToken_Applied(t *testing.T) {
	t.Parallel()
	n, err := nn.New[float64](
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](3, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithVisualizationEndpoint[float64](":0"),
		nn.WithVisualizationToken[float64]("secret"),
	)
	if err != nil {
		t.Fatalf("New with token: %v", err)
	}
	if err := n.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestWithVisualizationCORS_Applied(t *testing.T) {
	t.Parallel()
	n, err := nn.New[float64](
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](3, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithLoss[float64](loss.MSE),
		nn.WithVisualizationEndpoint[float64](":0"),
		nn.WithVisualizationCORS[float64](true),
	)
	if err != nil {
		t.Fatalf("New with CORS: %v", err)
	}
	if err := n.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

// ============================================================================
// NN.Close and TopologyVersion
// ============================================================================

func TestClose_NoVisServer_IsNoOp(t *testing.T) {
	t.Parallel()
	n := buildImmutable(t)
	if err := n.Close(); err != nil {
		t.Errorf("Close on NN without vis server: %v", err)
	}
}

func TestTopologyVersion_ImmutableIsZero(t *testing.T) {
	t.Parallel()
	n := buildImmutable(t)
	if v := n.TopologyVersion(); v != 0 {
		t.Errorf("Immutable TopologyVersion: want 0, got %d", v)
	}
}

func TestTopologyVersion_IncrementAfterMutation(t *testing.T) {
	t.Parallel()
	n := buildDynamic(t)
	before := n.TopologyVersion()
	if err := n.AddNeuron(0, 1); err != nil {
		t.Fatalf("AddNeuron: %v", err)
	}
	after := n.TopologyVersion()
	if after != before+1 {
		t.Errorf("TopologyVersion: want %d, got %d", before+1, after)
	}
}

// ============================================================================
// DYN-2 wrapper methods on NN[T]
// ============================================================================

func TestAddNeuron_Dynamic_Success(t *testing.T) {
	t.Parallel()
	n := buildDynamic(t)
	if err := n.AddNeuron(0, 2); err != nil {
		t.Errorf("AddNeuron: %v", err)
	}
}

func TestAddNeuron_Immutable_Error(t *testing.T) {
	t.Parallel()
	n := buildImmutable(t)
	if err := n.AddNeuron(0, 1); !errors.Is(err, utils.ErrImmutableMode) {
		t.Errorf("want ErrImmutableMode, got %v", err)
	}
}

func TestRemoveNeuron_Dynamic_Success(t *testing.T) {
	t.Parallel()
	n := buildDynamic(t)
	// Layer 0 has 4 cells; remove 1 → 3 remaining
	if err := n.RemoveNeuron(0, 1); err != nil {
		t.Errorf("RemoveNeuron: %v", err)
	}
}

func TestRemoveNeuron_Immutable_Error(t *testing.T) {
	t.Parallel()
	n := buildImmutable(t)
	if err := n.RemoveNeuron(0, 1); !errors.Is(err, utils.ErrImmutableMode) {
		t.Errorf("want ErrImmutableMode, got %v", err)
	}
}

func TestAddHiddenLayer_Dynamic_Success(t *testing.T) {
	t.Parallel()
	n := buildDynamic(t)
	if err := n.AddHiddenLayer(1, 3, activation.ReLU, false); err != nil {
		t.Errorf("AddHiddenLayer: %v", err)
	}
}

func TestAddHiddenLayer_Immutable_Error(t *testing.T) {
	t.Parallel()
	n := buildImmutable(t)
	if err := n.AddHiddenLayer(1, 3, activation.ReLU, false); !errors.Is(err, utils.ErrImmutableMode) {
		t.Errorf("want ErrImmutableMode, got %v", err)
	}
}

func TestRemoveHiddenLayer_Dynamic_Success(t *testing.T) {
	t.Parallel()
	n := buildDynamic(t) // 2 hidden layers → remove one
	if err := n.RemoveHiddenLayer(1); err != nil {
		t.Errorf("RemoveHiddenLayer: %v", err)
	}
}

func TestRemoveHiddenLayer_Immutable_Error(t *testing.T) {
	t.Parallel()
	n := buildImmutable(t)
	if err := n.RemoveHiddenLayer(0); !errors.Is(err, utils.ErrImmutableMode) {
		t.Errorf("want ErrImmutableMode, got %v", err)
	}
}

// ============================================================================
// Builder WithTopologyMode path
// ============================================================================

func TestBuilderWithTopologyMode(t *testing.T) {
	t.Parallel()
	n, err := nn.NewBuilder[float64]().
		Input(3).
		Dense(4, activation.SIGMOID, false).
		Output(2, activation.SIGMOID, false).
		WithTopologyMode(network.Dynamic).
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if err := n.AddNeuron(0, 1); err != nil {
		t.Errorf("AddNeuron on Builder Dynamic: %v", err)
	}
}
