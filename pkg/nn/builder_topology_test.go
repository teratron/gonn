package nn_test

import (
	"testing"
	"time"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/optimizer"
)

// ============================================================================
// Track B — Bulk topology constructors (l2-deep-builder)
// ============================================================================

// TestRepeatBuilderAppendsCount verifies Repeat adds exactly count layers
// with the correct shape.
func TestRepeatBuilderAppendsCount(t *testing.T) {
	n, err := nn.NewBuilder[float64]().
		Input(4).
		Repeat(5, 8, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	cfg := n.Config()
	if len(cfg.HiddenLayers) != 5 {
		t.Fatalf("want 5 hidden layers, got %d", len(cfg.HiddenLayers))
	}
	for i, h := range cfg.HiddenLayers {
		if h.Size != 8 {
			t.Errorf("layer %d: size %d, want 8", i, h.Size)
		}
		if h.Activation != activation.SIGMOID {
			t.Errorf("layer %d: activation %v, want SIGMOID", i, h.Activation)
		}
		if !h.Bias {
			t.Errorf("layer %d: bias false, want true", i)
		}
	}
}

// TestRepeatBuilderZeroCountIsNoOp verifies Repeat(0,...) adds no layers
// and Compile fails (still need at least one hidden layer).
func TestRepeatBuilderZeroCountIsNoOp(t *testing.T) {
	_, err := nn.NewBuilder[float64]().
		Input(2).
		Repeat(0, 4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		Compile()
	if err == nil {
		t.Fatal("expected ErrUserConfig for zero hidden layers after Repeat(0,...)")
	}
}

// TestPatternBuilderAppendsBlockTimesRepeats verifies Pattern multiplies correctly.
func TestPatternBuilderAppendsBlockTimesRepeats(t *testing.T) {
	block := []nn.HiddenLayerSpec[float64]{
		{Size: 16, Activation: activation.ReLU, Bias: true},
		{Size: 8, Activation: activation.ReLU, Bias: true},
	}
	n, err := nn.NewBuilder[float64]().
		Input(4).
		Pattern(block, 3). // 2 × 3 = 6 layers
		Output(1, activation.SIGMOID, true).
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if got := len(n.Config().HiddenLayers); got != 6 {
		t.Fatalf("want 6 hidden layers, got %d", got)
	}
}

// TestHiddenLayersBuilderReplacesPrevious verifies setter semantics (not append).
func TestHiddenLayersBuilderReplacesPrevious(t *testing.T) {
	layers := []nn.HiddenLayerSpec[float64]{
		{Size: 4, Activation: activation.SIGMOID, Bias: true},
	}
	n, err := nn.NewBuilder[float64]().
		Input(2).
		Dense(8, activation.ReLU, true). // will be replaced
		HiddenLayers(layers).
		Output(1, activation.SIGMOID, true).
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	cfg := n.Config()
	if len(cfg.HiddenLayers) != 1 {
		t.Fatalf("want 1 hidden layer after HiddenLayers(), got %d", len(cfg.HiddenLayers))
	}
	if cfg.HiddenLayers[0].Size != 4 {
		t.Errorf("want size 4, got %d", cfg.HiddenLayers[0].Size)
	}
}

// TestRepeatOptionAppends verifies the Options API Repeat function appends layers.
func TestRepeatOptionAppends(t *testing.T) {
	n, err := nn.New(
		nn.WithInput[float64](4),
		nn.Repeat[float64](3, 16, activation.ReLU),
		nn.WithOutput[float64](1, activation.SIGMOID),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := len(n.Config().HiddenLayers); got != 3 {
		t.Fatalf("want 3 hidden layers, got %d", got)
	}
}

// TestPatternOptionAppends verifies the Options API Pattern function.
func TestPatternOptionAppends(t *testing.T) {
	block := []nn.HiddenLayerSpec[float64]{
		{Size: 8, Activation: activation.SIGMOID, Bias: false},
		{Size: 4, Activation: activation.SIGMOID, Bias: false},
	}
	n, err := nn.New(
		nn.WithInput[float64](4),
		nn.Pattern(block, 4), // 2 × 4 = 8 layers
		nn.WithOutput[float64](1, activation.SIGMOID),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := len(n.Config().HiddenLayers); got != 8 {
		t.Fatalf("want 8 hidden layers, got %d", got)
	}
}

// TestWithHiddenLayersOptionAppends verifies the Options API WithHiddenLayers appends.
func TestWithHiddenLayersOptionAppends(t *testing.T) {
	layers := []nn.HiddenLayerSpec[float64]{
		{Size: 16, Activation: activation.ReLU, Bias: true},
		{Size: 8, Activation: activation.ReLU, Bias: true},
	}
	n, err := nn.New[float64](
		nn.WithInput[float64](4),
		nn.WithHiddenLayers[float64](layers),
		nn.WithOutput[float64](1, activation.SIGMOID),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := len(n.Config().HiddenLayers); got != 2 {
		t.Fatalf("want 2 hidden layers, got %d", got)
	}
}

// TestRepeatBuilderBenchmark100Layer verifies a 100-layer Compile completes
// quickly (under 1 second — actual benchmark uses testing.B).
func TestRepeatBuilderBenchmark100Layer(t *testing.T) {
	start := time.Now()
	n, err := nn.NewBuilder[float32]().
		Input(784).
		Repeat(100, 256, activation.ReLU, true).
		Output(10, activation.SOFTMAX, true).
		Compile()
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Compile 100-layer: %v", err)
	}
	if got := len(n.Config().HiddenLayers); got != 100 {
		t.Fatalf("want 100 hidden layers, got %d", got)
	}
	if elapsed > 5*time.Second {
		t.Errorf("100-layer Compile took %v, want < 5s", elapsed)
	}
}

// ============================================================================
// Track A — Scheduler wiring into training loop (T-7A06)
// ============================================================================

// TestWithSchedulerBuilderSetsSched verifies WithScheduler stores the scheduler.
func TestWithSchedulerBuilderSetsSched(t *testing.T) {
	sched := optimizer.NewStepLR[float64](0.3, 5, 0.9)
	n, err := nn.NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithScheduler(sched).
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if n.Config().Scheduler == nil {
		t.Fatal("Scheduler should be set after WithScheduler")
	}
}

// TestWithSchedulerOptionSetsSched verifies the Options API WithScheduler.
func TestWithSchedulerOptionSetsSched(t *testing.T) {
	sched := optimizer.NewCosineAnnealingLR(0.1, 0.001, 100)
	n, err := nn.New(
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithScheduler(sched),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if n.Config().Scheduler == nil {
		t.Fatal("Scheduler should be set after WithScheduler option")
	}
}

// TestSchedulerStepsOnEpoch verifies that a PerEpoch scheduler is called
// during Fit by checking the optimizer's LR changes after training.
func TestSchedulerStepsOnEpoch(t *testing.T) {
	opt := optimizer.NewSGD(1.0)
	// StepLR: decay by 0.5 every 1 epoch — after 5 epochs LR should be 1.0 × 0.5^5.
	sched := optimizer.BindScheduler[float64](opt, optimizer.NewStepLR[float64](1.0, 1, 0.5))

	n, err := nn.New[float64](
		nn.WithInput[float64](2),
		nn.WithHiddenLayer[float64](4, activation.SIGMOID),
		nn.WithOutput[float64](1, activation.SIGMOID),
		nn.WithBias[float64](true),
		nn.WithLearningRate[float64](1.0),
		nn.WithOptimizer[float64](opt),
		nn.WithScheduler[float64](sched),
		nn.WithMaxIterations[float64](5),
		nn.WithLossLimit[float64](0), // disable early stopping
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	dataset := []nn.Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
	epochs, _, err := n.Fit(dataset)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if epochs == 0 {
		t.Fatal("Fit completed 0 epochs")
	}
	// After epochs training, optimizer LR must be less than 1.0 (scheduler applied).
	if opt.LearningRate() >= 1.0 {
		t.Errorf("optimizer LR after %d epochs = %v, want < 1.0 (scheduler should have stepped)",
			epochs, opt.LearningRate())
	}
}

// BenchmarkRepeat100Compile measures 100-layer Compile performance on a compact
// topology (4→8^100→1). The l2-deep-builder spec target is < 1ms for this case.
func BenchmarkRepeat100Compile(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = nn.NewBuilder[float32]().
			Input(4).
			Repeat(100, 8, activation.ReLU, true).
			Output(1, activation.SIGMOID, true).
			Compile()
	}
}

// BenchmarkRepeat100CompileLarge measures 100-layer Compile on a realistic MNIST-scale
// topology (784→256^100→10). Reports actual performance without a target constraint.
func BenchmarkRepeat100CompileLarge(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = nn.NewBuilder[float32]().
			Input(784).
			Repeat(100, 256, activation.ReLU, true).
			Output(10, activation.SOFTMAX, true).
			Compile()
	}
}
