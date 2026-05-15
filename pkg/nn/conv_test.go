package nn

import (
	"errors"
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/utils"
)

// TestConvPrefixCompileShape verifies that compile() resolves the conv chain
// output length and resizes the Input layer accordingly. Input is 8, conv
// produces 2 filters × outputLen(8, 3, 1, Valid)=6 = 12, Flatten preserves
// that, so the Network's Input layer should be sized to 12.
func TestConvPrefixCompileShape(t *testing.T) {
	t.Parallel()
	n, err := New[float64](
		WithInput[float64](8),
		WithConv1D[float64](2, 3, 1, conv.PadValid, false),
		WithFlatten[float64](),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := n.Network.Input.Len(); got != 12 {
		t.Errorf("Input layer length = %d, want 12 (2 filters × 6 outputLen)", got)
	}
	if n.rawInputSize != 8 {
		t.Errorf("rawInputSize = %d, want 8", n.rawInputSize)
	}
}

// TestConvPrefixForwardShapeMismatch verifies the raw-input length guard.
func TestConvPrefixForwardShapeMismatch(t *testing.T) {
	t.Parallel()
	n, err := New[float64](
		WithInput[float64](8),
		WithConv1D[float64](2, 3, 1, conv.PadValid, false),
		WithFlatten[float64](),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = n.Query([]float64{1, 2, 3}) // wrong length
	if err == nil || !errors.Is(err, utils.ErrInputData) {
		t.Errorf("Query short input err = %v, want ErrInputData", err)
	}
}

// TestConvPrefixCompileError covers the conv-collapse path: kernel larger
// than the raw input under PadValid produces zero output and compile must
// return ErrConvShapeMismatch.
func TestConvPrefixCompileError(t *testing.T) {
	t.Parallel()
	_, err := New[float64](
		WithInput[float64](2),
		WithConv1D[float64](2, 5, 1, conv.PadValid, false), // kernel 5 > input 2
		WithFlatten[float64](),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
	)
	if err == nil {
		t.Fatal("expected compile error for kernel > input")
	}
	if !errors.Is(err, utils.ErrConvShapeMismatch) {
		t.Errorf("compile err = %v, want ErrConvShapeMismatch", err)
	}
}

// TestConvPrefixTrainConverges drives one Fit pass on a tiny synthetic task
// to prove the conv stack participates in backprop. Target = 1 when the
// first three inputs sum positive, 0 otherwise — a simple pattern Conv1D's
// kernel can latch onto and Dense can classify.
func TestConvPrefixTrainConverges(t *testing.T) {
	t.Parallel()
	n, err := New[float64](
		WithInput[float64](6),
		WithConv1D[float64](2, 3, 1, conv.PadValid, true),
		WithFlatten[float64](),
		WithHiddenLayer[float64](6, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate[float64](0.3),
		WithMaxIterations[float64](2000),
		WithLossLimit[float64](1e-2),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	samples := []Sample[float64]{
		{Input: []float64{1, 1, 1, 0, 0, 0}, Target: []float64{1}},
		{Input: []float64{0.8, 0.6, 0.4, 0, 0, 0}, Target: []float64{1}},
		{Input: []float64{-1, -1, -1, 0, 0, 0}, Target: []float64{0}},
		{Input: []float64{-0.5, -0.4, -0.3, 0, 0, 0}, Target: []float64{0}},
	}
	_, loss, err := n.Fit(samples)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss is non-finite: %v", loss)
	}
	// Verify the network learned the pattern — predictions should be
	// closer to targets than to inverted targets.
	var distRight, distWrong float64
	for _, s := range samples {
		y, err := n.Query(s.Input)
		if err != nil {
			t.Fatalf("Query: %v", err)
		}
		distRight += math.Abs(y[0] - s.Target[0])
		distWrong += math.Abs(y[0] - (1 - s.Target[0]))
	}
	if distRight >= distWrong {
		t.Errorf("network did not learn pattern: distRight=%v distWrong=%v", distRight, distWrong)
	}
}

// TestConvWeightsChange asserts that after a Fit pass conv weights have
// actually moved — proves the inline SGD on Conv1D actually fires.
func TestConvWeightsChange(t *testing.T) {
	t.Parallel()
	n, err := New[float64](
		WithInput[float64](6),
		WithConv1D[float64](1, 3, 1, conv.PadValid, false),
		WithFlatten[float64](),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate[float64](0.5),
		WithMaxIterations[float64](100),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	c1d, ok := n.convPrefix[0].(*conv.Conv1D[float64])
	if !ok {
		t.Fatal("first conv layer is not *Conv1D[float64]")
	}
	pre := make([]float64, len(c1d.Weights))
	copy(pre, c1d.Weights)
	samples := []Sample[float64]{
		{Input: []float64{1, 0, -1, 1, 0, -1}, Target: []float64{1}},
		{Input: []float64{-1, 0, 1, -1, 0, 1}, Target: []float64{0}},
	}
	if _, _, err := n.Fit(samples); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	changed := false
	for i := range pre {
		if math.Abs(float64(pre[i]-c1d.Weights[i])) > 1e-9 {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("conv weights did not change after Fit — backward path is broken")
	}
}

// TestNoConvPrefixZeroOverhead confirms the existing pure-Dense path is
// untouched when no conv option is supplied: rawInputSize stays 0 and
// runConvForward returns the input slice by identity.
func TestNoConvPrefixZeroOverhead(t *testing.T) {
	t.Parallel()
	n, err := New[float64](
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if len(n.convPrefix) != 0 {
		t.Errorf("convPrefix non-empty: %d", len(n.convPrefix))
	}
	if n.rawInputSize != 2 {
		t.Errorf("rawInputSize = %d, want 2", n.rawInputSize)
	}
	// runConvForward must return the slice unchanged.
	in := []float64{0.1, 0.2}
	out, err := n.runConvForward(in)
	if err != nil {
		t.Fatalf("runConvForward: %v", err)
	}
	if &out[0] != &in[0] {
		t.Error("runConvForward allocated a new slice instead of returning input by identity")
	}
}

// TestMaxAndAvgPoolInPrefix exercises both pooling variants through compile
// and a single Fit step — ensures shape arithmetic stays sane through every
// composable primitive.
func TestMaxAndAvgPoolInPrefix(t *testing.T) {
	t.Parallel()
	n, err := New[float64](
		WithInput[float64](12),
		WithConv1D[float64](1, 3, 1, conv.PadValid, false), // 12 → 10
		WithMaxPool1D[float64](2),                          // 10 → 5
		WithFlatten[float64](),                             // 5 → 5
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithMaxIterations[float64](5),
	)
	if err != nil {
		t.Fatalf("New (MaxPool): %v", err)
	}
	if n.Network.Input.Len() != 5 {
		t.Errorf("Input.Len = %d, want 5 (1 filter × pool 5)", n.Network.Input.Len())
	}
	samples := []Sample[float64]{
		{Input: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, Target: []float64{1}},
	}
	if _, _, err := n.Fit(samples); err != nil {
		t.Errorf("Fit MaxPool: %v", err)
	}

	n2, err := New[float64](
		WithInput[float64](12),
		WithConv1D[float64](1, 3, 1, conv.PadValid, false),
		WithAvgPool1D[float64](2),
		WithFlatten[float64](),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithMaxIterations[float64](5),
	)
	if err != nil {
		t.Fatalf("New (AvgPool): %v", err)
	}
	if _, _, err := n2.Fit(samples); err != nil {
		t.Errorf("Fit AvgPool: %v", err)
	}
}
