package nn

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// captureWarnings swaps utils.Logger for a buffer-backed slog.Logger
// so a single test can assert on Warn-level emissions, then restores
// the previous Logger via the returned cleanup. Tests using this helper
// must NOT run with t.Parallel() because utils.Logger is package-global.
func captureWarnings(t *testing.T) (buf *bytes.Buffer, restore func()) {
	t.Helper()
	buf = &bytes.Buffer{}
	prev := utils.Logger
	utils.Logger = slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	return buf, func() { utils.Logger = prev }
}

// TestDeepStackRandomInitWarn confirms emitSoftWarnings fires the
// [l1-neural-network-architecture] §6 advisory when len(HiddenLayers) > 5
// and WeightInit == Random. Phase 5 / Track B (T-5B02): with the gate
// lifted, a deep stack now actually compiles, so the warning becomes
// reachable and we exercise it end-to-end.
func TestDeepStackRandomInitWarn(t *testing.T) {
	buf, restore := captureWarnings(t)
	defer restore()

	const depth = 7
	opts := []Option[float64]{
		WithInput[float64](2),
		WithBias[float64](true),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate(0.1),
		WithLoss[float64](loss.MSE),
		WithWeightInit[float64](WeightInitRandom),
	}
	for range depth {
		opts = append(opts, WithHiddenLayer[float64](4, activation.SIGMOID))
	}
	if _, err := New(opts...); err != nil {
		t.Fatalf("deep-stack Compile must succeed in v0.6, got %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "deep stack") || !strings.Contains(out, "WeightInitRandom") {
		t.Errorf("expected deep-stack Random-init Warn in log, got:\n%s", out)
	}
	if !strings.Contains(out, "level=WARN") {
		t.Errorf("expected level=WARN entry in log, got:\n%s", out)
	}
}

// TestDeepStackXavierNoWarn is the negative control for the previous
// test: a 7-deep stack with Xavier init must NOT produce the deep-stack
// warning, ensuring the advisory is gated on WeightInit == Random.
func TestDeepStackXavierNoWarn(t *testing.T) {
	buf, restore := captureWarnings(t)
	defer restore()

	const depth = 7
	opts := []Option[float64]{
		WithInput[float64](2),
		WithBias[float64](true),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate(0.1),
		WithLoss[float64](loss.MSE),
		WithWeightInit[float64](WeightInitXavier),
	}
	for range depth {
		opts = append(opts, WithHiddenLayer[float64](4, activation.SIGMOID))
	}
	if _, err := New(opts...); err != nil {
		t.Fatalf("deep-stack Compile must succeed, got %v", err)
	}
	if strings.Contains(buf.String(), "deep stack") {
		t.Errorf("Xavier deep stack must NOT trigger the deep-stack warning, got:\n%s", buf.String())
	}
}

// xorSamples returns the canonical four-point XOR dataset shared by
// every smoke test below. Named samples keep individual test bodies
// short and readable.
func xorSamples() []Sample[float64] {
	return []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
}

// TestCompileFitMultiHiddenChainDepths checks T-5B03 against three
// chain depths (2, 3, 7) on the canonical XOR task. We assert that
// Compile succeeds and Fit runs to completion without error — full
// convergence at depth 7 with random init is not guaranteed in a
// short epoch budget, so the assertion is "no error", not "loss low".
// The single-hidden XOR convergence test in pkg/network is the
// regression baseline for actual learning.
func TestCompileFitMultiHiddenChainDepths(t *testing.T) {
	t.Parallel()
	depths := []int{2, 3, 7}
	for _, depth := range depths {
		t.Run("depth_"+itoa(depth), func(t *testing.T) {
			t.Parallel()
			opts := []Option[float64]{
				WithInput[float64](2),
				WithBias[float64](true),
				WithOutput[float64](1, activation.SIGMOID),
				WithLearningRate(0.3),
				WithLoss[float64](loss.MSE),
				WithWeightInit[float64](WeightInitXavier),
				WithMaxIterations[float64](200),
			}
			for range depth {
				opts = append(opts, WithHiddenLayer[float64](4, activation.SIGMOID))
			}
			n, err := New(opts...)
			if err != nil {
				t.Fatalf("depth=%d Compile: %v", depth, err)
			}
			if n.State() != stateOperational {
				t.Errorf("depth=%d post-Compile state = %v; want Operational", depth, n.State())
			}
			epochs, _, err := n.Fit(xorSamples())
			if err != nil {
				t.Fatalf("depth=%d Fit: %v", depth, err)
			}
			if epochs == 0 {
				t.Errorf("depth=%d Fit completed 0 epochs; want > 0", depth)
			}
		})
	}
}

// TestCompileMultiHiddenTwoHiddenConverges sanity-checks that a 2-hidden
// XOR network with Xavier init still learns end-to-end through the
// public facade. This complements pkg/network's golden-math tests with
// a Compile→Fit→Query loop measured at the user-visible API.
func TestCompileMultiHiddenTwoHiddenConverges(t *testing.T) {
	t.Parallel()
	n, err := New(
		WithInput[float64](2),
		WithBias[float64](true),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate(0.5),
		WithLoss[float64](loss.MSE),
		WithWeightInit[float64](WeightInitXavier),
		WithMaxIterations[float64](50000),
		WithLossLimit(0.05),
	)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	_, finalLoss, err := n.Fit(xorSamples())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// Xavier uniform on a 2-4-4-1 network uses larger initial weights than
	// the old U[-0.5,0.5] fallback (T-6B06 debt fix), so more epochs are
	// needed to confirm convergence.
	const target = 0.10
	if float64(finalLoss) > target {
		t.Errorf("2-hidden XOR loss %v above tolerance %v after 50000 epochs", finalLoss, target)
	}
}

// itoa avoids strconv import for one tiny use-case.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [4]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
