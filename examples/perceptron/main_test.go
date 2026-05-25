package main

import (
	"bytes"
	"errors"
	"io"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// ============================================================================
// dataSet()
// ============================================================================

// TestDataSetExactValues verifies the canonical E03 stream is byte-identical
// to the spec table. Any accidental edit of the literal would break the
// sliding-window math downstream.
func TestDataSetExactValues(t *testing.T) {
	t.Parallel()
	want := []float32{.27, -.31, -.52, .66, .81, -.13, .2, .49, .11, -.73, .28}
	got := dataSet()
	if len(got) != len(want) {
		t.Fatalf("dataSet() len = %d, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("dataSet()[%d] = %v, want %v", i, got[i], w)
		}
	}
}

// TestDataSetIsReproducible verifies that repeated calls to dataSet() return
// identical values — the function must not carry mutable state.
func TestDataSetIsReproducible(t *testing.T) {
	t.Parallel()
	a, b := dataSet(), dataSet()
	if len(a) != len(b) {
		t.Fatalf("dataSet() returned different lengths: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("dataSet()[%d]: first=%v second=%v", i, a[i], b[i])
		}
	}
}

// TestDataSetSlidingWindowCount verifies that the data contains exactly the
// number of valid (input, target) windows that trainPerceptron will consume.
// Windows span [i-lenInput : i] → [i : i+lenOutput] for i in [lenInput, len-lenOutput].
func TestDataSetSlidingWindowCount(t *testing.T) {
	t.Parallel()
	data := dataSet()
	lenData := len(data) - lenOutput
	var count int
	for i := lenInput; i <= lenData; i++ {
		count++
	}
	// For an 11-element stream with window (3 in, 2 out) the valid positions
	// are i = 3..9, giving 7 training samples.
	if count != 7 {
		t.Errorf("sliding window count = %d, want 7", count)
	}
}

// ============================================================================
// build() — topology and hyperparameters
// ============================================================================

// TestBuildCompiles verifies the spec-canonical 4-hidden perceptron
// topology (3 → Sigmoid(5) → ReLU(10) → Sigmoid(5) → SoftMax(2))
// compiles cleanly under v0.2. Catches Compile() regressions on the
// multi-hidden seam without paying for the full training run.
func TestBuildCompiles(t *testing.T) {
	if _, err := build(); err != nil {
		t.Fatalf("build: %v", err)
	}
}

// TestBuildStateIsOperational verifies that build() transitions the network
// to Operational state (required before Train/Query can be called).
func TestBuildStateIsOperational(t *testing.T) {
	t.Parallel()
	n, err := build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if got := n.State().String(); got != "Operational" {
		t.Errorf("State() = %q, want \"Operational\"", got)
	}
}

// TestBuildTopology verifies the canonical 3 → Sigmoid(5) → ReLU(10) →
// Sigmoid(5, no-bias) → SoftMax(2) topology from the E03 spec. Mixed bias
// is the distinguishing feature — the pre-output Sigmoid layer carries no bias.
func TestBuildTopology(t *testing.T) {
	t.Parallel()
	n, err := build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	cfg := n.Config()

	if cfg.InputSize != uint(lenInput) {
		t.Errorf("InputSize = %d, want %d", cfg.InputSize, lenInput)
	}
	if cfg.OutputSize != uint(lenOutput) {
		t.Errorf("OutputSize = %d, want %d", cfg.OutputSize, lenOutput)
	}
	if got := len(cfg.HiddenLayers); got != 3 {
		t.Fatalf("len(HiddenLayers) = %d, want 3", got)
	}

	layers := []struct {
		size uint
		act  activation.Type
		bias bool
	}{
		{5, activation.SIGMOID, true},
		{10, activation.ReLU, true},
		{5, activation.SIGMOID, false}, // spec: no bias on pre-output Sigmoid
	}
	for i, want := range layers {
		h := cfg.HiddenLayers[i]
		if h.Size != want.size {
			t.Errorf("HiddenLayers[%d].Size = %d, want %d", i, h.Size, want.size)
		}
		if h.Activation != want.act {
			t.Errorf("HiddenLayers[%d].Activation = %v, want %v", i, h.Activation, want.act)
		}
		if h.Bias != want.bias {
			t.Errorf("HiddenLayers[%d].Bias = %v, want %v", i, h.Bias, want.bias)
		}
	}

	if cfg.OutputActivation != activation.Linear {
		t.Errorf("OutputActivation = %v, want Linear", cfg.OutputActivation)
	}
	if !cfg.OutputBias {
		t.Error("OutputBias = false, want true")
	}
}

// TestBuildHyperparameters verifies that all training-control fields are
// populated exactly as documented in the E03 spec comments.
func TestBuildHyperparameters(t *testing.T) {
	t.Parallel()
	n, err := build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	cfg := n.Config()

	if cfg.LearningRate != 0.3 {
		t.Errorf("LearningRate = %v, want 0.3", cfg.LearningRate)
	}
	if cfg.MaxIterations != 100_000 {
		t.Errorf("MaxIterations = %d, want 100000", cfg.MaxIterations)
	}
	if cfg.LossLimit != 1e-6 {
		t.Errorf("LossLimit = %v, want 1e-6", cfg.LossLimit)
	}
	if cfg.LossType != loss.ARCTAN {
		t.Errorf("LossType = %v, want ARCTAN", cfg.LossType)
	}
	if cfg.WeightInit != nn.WeightInitXavier {
		t.Errorf("WeightInit = %v, want %v", cfg.WeightInit, nn.WeightInitXavier)
	}
}

// ============================================================================
// Query contract on a compiled-but-untrained network
// ============================================================================

// TestQueryOnUntrainedNetShape verifies that a freshly compiled (untrained)
// network accepts the reference input and returns a slice of the right length.
// Ensures Compile() alone is sufficient for inference without panicking.
func TestQueryOnUntrainedNetShape(t *testing.T) {
	t.Parallel()
	n, err := build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	out, err := n.Query([]float32{-.52, .66, .81})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != lenOutput {
		t.Errorf("len(Query output) = %d, want %d", len(out), lenOutput)
	}
}

// TestQueryRejectsWrongInputLen verifies that Query returns ErrInputData when
// the caller supplies a slice whose length differs from lenInput.
func TestQueryRejectsWrongInputLen(t *testing.T) {
	t.Parallel()
	n, err := build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	_, err = n.Query([]float32{1, 2, 3, 4}) // lenInput=3; supplying 4 is wrong
	if !errors.Is(err, utils.ErrInputData) {
		t.Errorf("Query with wrong input length must return ErrInputData, got %v", err)
	}
}

// ============================================================================
// Trained-network behaviour
// ============================================================================

// TestTrainProducesQuery runs an abbreviated training loop (the build()
// configuration caps at 5000 epochs in trainPerceptron — full 100k is
// for the binary). The smoke test asserts only that:
//
//   - training completes without error;
//   - the post-train query has the expected output shape (lenOutput).
//
// Numeric proximity to the published reference is intentionally NOT
// asserted — random init plus the spec's loose convergence target make
// per-element drift dependent on PCG seed. Convergence sanity is the
// pkg/network XOR test's job; this example documents API usage.
func TestTrainProducesQuery(t *testing.T) {
	n, err := build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	pred, err := trainPerceptron(n)
	if err != nil {
		t.Fatalf("trainPerceptron: %v", err)
	}
	if len(pred) != lenOutput {
		t.Errorf("Query output len = %d, want %d", len(pred), lenOutput)
	}
}

// TestTrainedPerceptron runs a single full training cycle and then asserts
// multiple properties of the result. Consolidating into one function avoids
// three separate 100k-epoch loops.
//
// Skipped when -short is set — 100k × 7 Train calls are too slow for CI
// quick checks.
func TestTrainedPerceptron(t *testing.T) {
	if testing.Short() {
		t.Skip("100k-epoch training loop skipped in -short mode")
	}

	n, err := build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	out, err := trainPerceptron(n)
	if err != nil {
		t.Fatalf("trainPerceptron: %v", err)
	}
	t.Logf("Query([-0.52, 0.66, 0.81]) = %v", out)

	// output_len: already covered by TestTrainProducesQuery, but asserting
	// here guards against a future regression where trainPerceptron returns
	// the wrong slice without failing.
	t.Run("output_len", func(t *testing.T) {
		if len(out) != lenOutput {
			t.Errorf("len = %d, want %d", len(out), lenOutput)
		}
	})

	// output_bounds: Linear output has no range constraint, but after
	// training on targets [-0.13, 0.2] the values must be finite and
	// reasonably close to the reference window (loose tolerance ±0.5).
	t.Run("output_bounds", func(t *testing.T) {
		ref := []float32{-.13, .2}
		for i, v := range out {
			if diff := float64(v - ref[i]); diff < -0.5 || diff > 0.5 {
				t.Errorf("output[%d] = %v too far from reference %v (tolerance ±0.5)", i, v, ref[i])
			}
		}
	})

	// output_finite: NaN or Inf would indicate a weight explosion or a
	// zero-division in the ARCTAN loss path.
	t.Run("output_finite", func(t *testing.T) {
		for i, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("output[%d] = %v is non-finite", i, v)
			}
		}
	})

	// query_deterministic: a forward pass has no stochastic components
	// (no dropout registered in build()), so two consecutive Query calls
	// on the same frozen weights must return bit-identical results.
	t.Run("query_deterministic", func(t *testing.T) {
		input := []float32{-.52, .66, .81}
		first, err := n.Query(input)
		if err != nil {
			t.Fatalf("first Query: %v", err)
		}
		second, err := n.Query(input)
		if err != nil {
			t.Fatalf("second Query: %v", err)
		}
		for i := range first {
			if first[i] != second[i] {
				t.Errorf("output[%d]: first=%v second=%v (must be identical)", i, first[i], second[i])
			}
		}
	})
}

// ============================================================================
// main() output
// ============================================================================

// captureStdout replaces os.Stdout with a pipe, calls f, then drains the pipe
// and returns the captured output. Not parallel-safe — uses a global fd.
func captureStdout(f func()) string {
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	f()
	w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	io.Copy(&buf, r) //nolint:errcheck
	return buf.String()
}

// TestMainOutput captures the stdout written by main() and asserts all three
// lines documented in the E03 spec:
//
//  1. "Elapsed: …"         — timing line confirms training completed.
//  2. "Query([-0.52, 0.66, 0.81]) = …" — result line includes the reference input.
//  3. "(reference ≈ [-0.13, 0.2])" — annotation from the format string literal.
//
// Skipped under -short because main() runs the full 100k-epoch training loop.
func TestMainOutput(t *testing.T) {
	if testing.Short() {
		t.Skip("main() runs 100k-epoch training loop; skipped in -short mode")
	}

	out := captureStdout(main)

	t.Run("prints_elapsed", func(t *testing.T) {
		if !strings.Contains(out, "Elapsed:") {
			t.Errorf("output missing \"Elapsed:\"; got:\n%s", out)
		}
	})

	t.Run("prints_query_header", func(t *testing.T) {
		if !strings.Contains(out, "Query([-0.52, 0.66, 0.81])") {
			t.Errorf("output missing query header; got:\n%s", out)
		}
	})

	t.Run("prints_reference_annotation", func(t *testing.T) {
		if !strings.Contains(out, "reference ≈ [-0.13, 0.2]") {
			t.Errorf("output missing reference annotation; got:\n%s", out)
		}
	})
}
