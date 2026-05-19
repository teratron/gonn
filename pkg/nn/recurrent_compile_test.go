package nn

import (
	"errors"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer/recurrent"
	"github.com/teratron/gonn/pkg/utils"
)

// TestRecurrentCompile verifies that WithGRU / WithLSTM / WithSimpleRNN /
// WithLastStep compose through compile() and resize the Input layer to the
// recurrent stack's final output (REC-9 composition rule).
func TestRecurrentCompile(t *testing.T) {
	t.Parallel()

	t.Run("gru_plus_laststep", func(t *testing.T) {
		t.Parallel()
		// seqLen=4, inSize=3 → input=12; GRU hidden=5 → output=20 (4*5);
		// LastStep(4,5) → output=5; Dense(8) → Output(1).
		n, err := New(
			WithInput[float64](12),
			WithGRU[float64](4, 3, 5),
			WithLastStep[float64](4, 5),
			WithHiddenLayer[float64](8, activation.ReLU),
			WithOutput[float64](1, activation.SIGMOID),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		// After compile the Input layer is sized to the conv-chain output (5).
		if got := n.Network.Input.Len(); got != 5 {
			t.Errorf("Input layer length = %d, want 5 (LastStep output)", got)
		}
		// rawInputSize must be the original declaration.
		if n.rawInputSize != 12 {
			t.Errorf("rawInputSize = %d, want 12", n.rawInputSize)
		}
		// convPrefix must hold both recurrent layers.
		if len(n.convPrefix) != 2 {
			t.Errorf("convPrefix length = %d, want 2", len(n.convPrefix))
		}
		_, ok0 := n.convPrefix[0].(*recurrent.GRU[float64])
		_, ok1 := n.convPrefix[1].(*recurrent.LastStep[float64])
		if !ok0 || !ok1 {
			t.Errorf("convPrefix types mismatch: [0]=%T [1]=%T", n.convPrefix[0], n.convPrefix[1])
		}
	})

	t.Run("lstm_no_laststep", func(t *testing.T) {
		t.Parallel()
		// seqLen=3, inSize=4 → input=12; LSTM hidden=6 → output=18 (3*6); Dense(8) → Output(1).
		n, err := New(
			WithInput[float64](12),
			WithLSTM[float64](3, 4, 6),
			WithHiddenLayer[float64](8, activation.TanH),
			WithOutput[float64](1, activation.SIGMOID),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := n.Network.Input.Len(); got != 18 {
			t.Errorf("Input layer length = %d, want 18 (seqLen*hidden)", got)
		}
		if n.rawInputSize != 12 {
			t.Errorf("rawInputSize = %d, want 12", n.rawInputSize)
		}
	})

	t.Run("simplernn_plus_laststep", func(t *testing.T) {
		t.Parallel()
		// seqLen=5, inSize=2 → input=10; SimpleRNN hidden=4 → output=20; LastStep(5,4) → 4.
		n, err := New(
			WithInput[float64](10),
			WithSimpleRNN[float64](5, 2, 4),
			WithLastStep[float64](5, 4),
			WithHiddenLayer[float64](4, activation.SIGMOID),
			WithOutput[float64](1, activation.SIGMOID),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := n.Network.Input.Len(); got != 4 {
			t.Errorf("Input layer length = %d, want 4", got)
		}
	})

	t.Run("shape_mismatch_gru_input", func(t *testing.T) {
		t.Parallel()
		// InputSize=10 but GRU expects seqLen=4 * inSize=3 = 12 → mismatch.
		_, err := New(
			WithInput[float64](10),
			WithGRU[float64](4, 3, 5),
			WithLastStep[float64](4, 5),
			WithHiddenLayer[float64](4, activation.SIGMOID),
			WithOutput[float64](1, activation.SIGMOID),
		)
		if err == nil {
			t.Fatal("expected error for shape mismatch, got nil")
		}
		if !errors.Is(err, utils.ErrUserConfig) {
			t.Errorf("err = %v, want wrapping ErrUserConfig", err)
		}
		if !errors.Is(err, utils.ErrRecurrentShapeMismatch) {
			t.Errorf("err = %v, want wrapping ErrRecurrentShapeMismatch", err)
		}
	})

	t.Run("shape_mismatch_laststep_hidden", func(t *testing.T) {
		t.Parallel()
		// GRU hidden=5 → output seqLen*5=20; LastStep expects seqLen*6=24 → mismatch.
		_, err := New(
			WithInput[float64](12),
			WithGRU[float64](4, 3, 5),
			WithLastStep[float64](4, 6),
			WithHiddenLayer[float64](4, activation.SIGMOID),
			WithOutput[float64](1, activation.SIGMOID),
		)
		if err == nil {
			t.Fatal("expected error for LastStep hidden mismatch, got nil")
		}
		if !errors.Is(err, utils.ErrRecurrentShapeMismatch) {
			t.Errorf("err = %v, want wrapping ErrRecurrentShapeMismatch", err)
		}
	})
}
