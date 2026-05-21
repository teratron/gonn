package nn

import (
	"errors"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// ============================================================================
// Builder API — Track A
// ============================================================================

func TestNewBuilderStartsConfiguring(t *testing.T) {
	t.Parallel()
	n := NewBuilder[float64]()
	if got := n.State(); got != stateConfiguring {
		t.Errorf("NewBuilder state = %v; want Configuring", got)
	}
}

func TestBuilderHappyPath(t *testing.T) {
	t.Parallel()
	n, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithLoss(loss.MSE).
		WithLearningRate(0.3).
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if n.State() != stateOperational {
		t.Errorf("post-Compile state = %v; want Operational", n.State())
	}
	if n.LearningRate != 0.3 {
		t.Errorf("LearningRate = %v; want 0.3", n.LearningRate)
	}
}

func TestBuilderHiddenAlias(t *testing.T) {
	t.Parallel()
	n := NewBuilder[float64]().Input(2).Hidden(3, activation.ReLU, false)
	if got := len(n.Config().HiddenLayers); got != 1 {
		t.Errorf("Hidden alias did not append a layer; got %d", got)
	}
}

func TestBuilderRejectsMissingInput(t *testing.T) {
	t.Parallel()
	_, err := NewBuilder[float64]().
		Dense(4, activation.SIGMOID, false).
		Output(1, activation.SIGMOID, false).
		Compile()
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig (InputMissing), got %v", err)
	}
}

func TestBuilderRejectsMissingOutput(t *testing.T) {
	t.Parallel()
	_, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, false).
		Compile()
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig (OutputMissing), got %v", err)
	}
}

func TestBuilderRejectsMissingHidden(t *testing.T) {
	t.Parallel()
	_, err := NewBuilder[float64]().
		Input(2).
		Output(1, activation.SIGMOID, false).
		Compile()
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig (no hidden), got %v", err)
	}
}

func TestBuilderRejectsZeroSize(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		fn   func() (*NN[float64], error)
	}{
		{
			"input zero",
			func() (*NN[float64], error) {
				return NewBuilder[float64]().Input(0).
					Dense(4, activation.SIGMOID, false).
					Output(1, activation.SIGMOID, false).Compile()
			},
		},
		{
			"hidden zero",
			func() (*NN[float64], error) {
				return NewBuilder[float64]().Input(2).
					Dense(0, activation.SIGMOID, false).
					Output(1, activation.SIGMOID, false).Compile()
			},
		},
		{
			"output zero",
			func() (*NN[float64], error) {
				return NewBuilder[float64]().Input(2).
					Dense(4, activation.SIGMOID, false).
					Output(0, activation.SIGMOID, false).Compile()
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			_, err := c.fn()
			if !errors.Is(err, utils.ErrUserConfig) {
				t.Errorf("expected ErrUserConfig, got %v", err)
			}
		})
	}
}

func TestBuilderAcceptsMultiHidden(t *testing.T) {
	t.Parallel()
	// Phase 5 / Track B (T-5B01) lifts the v0.5 single-hidden gate.
	// The Builder must now accept any positive HiddenLayers count and
	// produce an Operational network — multi-hidden is the v0.6 default.
	n, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).Compile()
	if err != nil {
		t.Fatalf("multi-hidden Builder Compile must succeed in v0.6, got %v", err)
	}
	if n.State() != stateOperational {
		t.Errorf("post-Compile state = %v; want Operational", n.State())
	}
}

func TestBuilderRejectsUnknownActivation(t *testing.T) {
	t.Parallel()
	_, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.Type(99), true).
		Output(1, activation.SIGMOID, true).Compile()
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig (UnknownActivation), got %v", err)
	}
}

func TestBuilderRejectsUnknownLoss(t *testing.T) {
	t.Parallel()
	_, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithLoss(loss.Type(99)).Compile()
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig (UnknownLoss), got %v", err)
	}
}

func TestBuilderRejectsUnknownWeightInit(t *testing.T) {
	t.Parallel()
	_, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithWeightInit(WeightInitMethod("bogus")).Compile()
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig (UnknownInit), got %v", err)
	}
}

func TestPostCompileMutationsAreNoOps(t *testing.T) {
	t.Parallel()
	n, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	// Capture pre-mutation config; the post-Compile call must not change it.
	before := n.Config()
	n.Input(99) // should be a no-op + Logger.Warn
	n.Dense(99, activation.ReLU, false)
	n.WithLearningRate(99)
	after := n.Config()
	if after.InputSize != before.InputSize {
		t.Errorf("post-Compile Input mutated InputSize: %d -> %d", before.InputSize, after.InputSize)
	}
	if len(after.HiddenLayers) != len(before.HiddenLayers) {
		t.Errorf("post-Compile Dense added a hidden layer: %d -> %d",
			len(before.HiddenLayers), len(after.HiddenLayers))
	}
	if after.LearningRate != before.LearningRate {
		t.Errorf("post-Compile WithLearningRate mutated rate: %v -> %v",
			before.LearningRate, after.LearningRate)
	}
}

func TestCompileTwiceIsIdempotent(t *testing.T) {
	t.Parallel()
	n, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if _, err := n.Compile(); !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("second Compile must return ErrUserConfig (already compiled), got %v", err)
	}
}

func TestMustCompilePanicsOnInvalid(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("MustCompile must panic on invalid config")
		}
	}()
	_ = NewBuilder[float64]().Input(2).MustCompile()
}

// ============================================================================
// Functional Options API — Track B
// ============================================================================

func TestOptionsAPIHappyPath(t *testing.T) {
	t.Parallel()
	n, err := New(
		WithInput[float64](2),
		WithBias[float64](true),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate[float64](0.3),
		WithLoss[float64](loss.MSE),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if n.State() != stateOperational {
		t.Errorf("post-New state = %v; want Operational", n.State())
	}
}

func TestMustNewPanicsOnInvalid(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("MustNew must panic on invalid config")
		}
	}()
	_ = MustNew(WithInput[float64](2))
}

func TestPresetXORCompiles(t *testing.T) {
	t.Parallel()
	n, err := New(PresetXOR[float64]())
	if err != nil {
		t.Fatalf("PresetXOR: %v", err)
	}
	if n.LearningRate != 0.3 {
		t.Errorf("PresetXOR rate = %v; want 0.3", n.LearningRate)
	}
}

func TestPresetMNISTCompiles(t *testing.T) {
	t.Parallel()
	// PresetMNIST is a 2-hidden classifier; Phase 5 / Track B (T-5B01)
	// lifts the gate that previously rejected it in v0.5. Smoke-only —
	// MNIST training is gated on the dataset-loader spec (E06).
	n, err := New(PresetMNIST[float64]())
	if err != nil {
		t.Fatalf("PresetMNIST Compile must succeed in v0.6, got %v", err)
	}
	if n.State() != stateOperational {
		t.Errorf("PresetMNIST post-Compile state = %v; want Operational", n.State())
	}
}

func TestPresetRegressionCompiles(t *testing.T) {
	t.Parallel()
	// PresetRegression also uses 2 hidden layers (size, size/2). Same
	// gate lift as MNIST — assert the surface compiles cleanly.
	n, err := New(
		WithInput[float64](4),
		WithOutput[float64](1, activation.Linear),
		PresetRegression[float64](4, 16),
	)
	if err != nil {
		t.Fatalf("PresetRegression Compile must succeed in v0.6, got %v", err)
	}
	if n.State() != stateOperational {
		t.Errorf("PresetRegression post-Compile state = %v; want Operational", n.State())
	}
}

func TestSequentialAppendsCount(t *testing.T) {
	t.Parallel()
	cfg := Config[float64]{}
	Sequential[float64](3, 5, activation.SIGMOID)(&cfg)
	if got := len(cfg.HiddenLayers); got != 3 {
		t.Errorf("Sequential(3) appended %d layers; want 3", got)
	}
}

func TestDeepNetworkHalvesWithFloor(t *testing.T) {
	t.Parallel()
	cfg := Config[float64]{}
	DeepNetwork[float64](16, 5, activation.ReLU)(&cfg)
	wantSizes := []uint{16, 8, 4, 2, 2}
	for i, h := range cfg.HiddenLayers {
		if h.Size != wantSizes[i] {
			t.Errorf("DeepNetwork[%d].Size = %d; want %d", i, h.Size, wantSizes[i])
		}
	}
}

func TestStandardSetupBundlesDefaults(t *testing.T) {
	t.Parallel()
	cfg := Config[float64]{}
	StandardSetup[float64](0.42)(&cfg)
	if cfg.LearningRate != 0.42 {
		t.Errorf("rate = %v; want 0.42", cfg.LearningRate)
	}
	if !cfg.DefaultBias {
		t.Errorf("DefaultBias must be true after StandardSetup")
	}
	if cfg.WeightInit != WeightInitXavier {
		t.Errorf("WeightInit = %v; want xavier", cfg.WeightInit)
	}
}

func TestBuilderAndOptionsConverge(t *testing.T) {
	t.Parallel()
	// Same XOR network via both styles must populate identical Config[T].
	a, err := NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithLearningRate(0.3).
		WithLoss(loss.MSE).Compile()
	if err != nil {
		t.Fatalf("Builder: %v", err)
	}
	b, err := New(
		WithInput[float64](2),
		WithBias[float64](true),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate[float64](0.3),
		WithLoss[float64](loss.MSE),
	)
	if err != nil {
		t.Fatalf("Options: %v", err)
	}
	ca, cb := a.Config(), b.Config()
	if ca.InputSize != cb.InputSize ||
		ca.OutputSize != cb.OutputSize ||
		ca.OutputActivation != cb.OutputActivation ||
		ca.LearningRate != cb.LearningRate ||
		ca.LossType != cb.LossType ||
		len(ca.HiddenLayers) != len(cb.HiddenLayers) {
		t.Errorf("style A vs B Config mismatch:\n  A=%+v\n  B=%+v", ca, cb)
	}
}

// ============================================================================
// Train / Query / Verify — Track C
// ============================================================================

func TestQueryRejectsUncompiled(t *testing.T) {
	t.Parallel()
	n := NewBuilder[float64]()
	_, err := n.Query([]float64{1, 2})
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("Query on uncompiled net must return ErrUserConfig, got %v", err)
	}
}

func TestQueryReadsForward(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	out, err := n.Query([]float64{0, 1})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != 1 {
		t.Errorf("len(Query output) = %d; want 1", len(out))
	}
	// Sigmoid output is in (0, 1) regardless of weights.
	if out[0] <= 0 || out[0] >= 1 {
		t.Errorf("sigmoid output %v outside (0, 1)", out[0])
	}
}

func TestVerifyDoesNotMutateWeights(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	before := n.snapshotWeights(nil)
	if _, err := n.Verify([]float64{0, 1}, []float64{1}); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	after := n.snapshotWeights(nil)
	if len(before) != len(after) {
		t.Fatalf("weight count changed: %d -> %d", len(before), len(after))
	}
	for i := range before {
		if before[i] != after[i] {
			t.Errorf("weight[%d] changed: %v -> %v", i, before[i], after[i])
		}
	}
}

func TestTrainRejectsUncompiled(t *testing.T) {
	t.Parallel()
	n := NewBuilder[float64]()
	_, err := n.Train([]float64{0, 1}, []float64{1})
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("Train on uncompiled must return ErrUserConfig, got %v", err)
	}
}

func TestFitConvergesXOR(t *testing.T) {
	t.Parallel()
	n := MustNew(
		PresetXOR[float64](),
		WithMaxIterations[float64](10_000),
		WithLossLimit[float64](0.02),
	)
	dataset := []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
	epochs, finalLoss, err := n.Fit(dataset)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if finalLoss > 0.05 {
		t.Errorf("Fit did not converge: epochs=%d, finalLoss=%v (want < 0.05)", epochs, finalLoss)
	}
	t.Logf("XOR via NN.Fit converged in %d epochs, finalLoss = %v", epochs, finalLoss)

	// Sanity: predictions must land on the right side of 0.5.
	for _, sample := range dataset {
		got, err := n.Query(sample.Input)
		if err != nil {
			t.Fatalf("Query: %v", err)
		}
		if math.Abs(got[0]-sample.Target[0]) > 0.4 {
			t.Errorf("XOR(%v) = %v; want ~%v", sample.Input, got[0], sample.Target[0])
		}
	}
}

func TestFitInvokesEpochCallback(t *testing.T) {
	t.Parallel()
	var callCount int
	var lastEpoch uint
	n := MustNew(
		PresetXOR[float64](),
		WithMaxIterations[float64](3),
		WithLossLimit[float64](-1), // unreachable — force max iterations
		WithEpochCallback[float64](func(epoch uint, lossValue float64) {
			callCount++
			lastEpoch = epoch
		}),
	)
	dataset := []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
	}
	if _, _, err := n.Fit(dataset); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if callCount != 3 {
		t.Errorf("EpochCallback called %d times; want 3", callCount)
	}
	if lastEpoch != 3 {
		t.Errorf("last epoch = %d; want 3", lastEpoch)
	}
}

func TestFitRejectsEmptyDataset(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	_, _, err := n.Fit(nil)
	if !errors.Is(err, utils.ErrInputData) {
		t.Errorf("Fit on empty dataset must return ErrInputData, got %v", err)
	}
}

// ============================================================================
// Lifecycle Control — Track D
// ============================================================================

func TestStopRequestsEarlyExit(t *testing.T) {
	t.Parallel()
	// Deterministic synchronisation barrier: the first epoch callback
	// signals "I have entered the loop" via started, then blocks on
	// stopReady until the test has issued Stop. This avoids the race
	// where Fit completes 100k tiny epochs in the time it takes the
	// test goroutine to be scheduled and observe controlRunning.
	started := make(chan struct{})
	stopReady := make(chan struct{})
	var firstEpoch sync.Once

	n := MustNew(
		PresetXOR[float64](),
		WithMaxIterations[float64](100_000),
		WithLossLimit[float64](-1),
		WithEpochCallback[float64](func(epoch uint, _ float64) {
			firstEpoch.Do(func() {
				close(started)
				<-stopReady
			})
		}),
	)
	dataset := []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}

	var wg sync.WaitGroup
	wg.Add(1)
	var epochs uint
	var fitErr error
	go func() {
		defer wg.Done()
		epochs, _, fitErr = n.Fit(dataset)
	}()

	<-started // worker is parked inside the first EpochCallback
	if err := n.Stop(); err != nil {
		t.Errorf("Stop: %v", err)
	}
	close(stopReady) // release the worker — next safe-point will see Stopped

	wg.Wait()
	if fitErr != nil {
		t.Errorf("Fit returned error after Stop: %v", fitErr)
	}
	if epochs >= 100_000 {
		t.Errorf("Stop did not interrupt training: epochs=%d", epochs)
	}
}

func TestPauseResumeCycle(t *testing.T) {
	t.Parallel()
	n := MustNew(
		PresetXOR[float64](),
		WithMaxIterations[float64](2000),
		WithLossLimit[float64](-1), // run full loop
	)
	dataset := []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
	}

	var wg sync.WaitGroup
	wg.Go(func() {
		_, _, _ = n.Fit(dataset)
	})

	// Spin until the worker has transitioned to Running. Avoids a flaky
	// time.Sleep race — on slow CI the worker may not have entered Fit
	// yet after a fixed 20ms wait.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if n.control.Load() == controlRunning {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if n.control.Load() != controlRunning {
		t.Fatalf("worker did not enter Running state within deadline")
	}

	if err := n.Pause(); err != nil {
		t.Fatalf("Pause: %v", err)
	}
	// Resume must succeed from Paused.
	if err := n.Resume(); err != nil {
		t.Errorf("Resume after Pause: %v", err)
	}
	if err := n.Stop(); err != nil {
		t.Errorf("Stop after Resume: %v", err)
	}
	wg.Wait()
}

func TestPauseFromIdleErrors(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	if err := n.Pause(); !errors.Is(err, utils.ErrControl) {
		t.Errorf("Pause on idle must return ErrControl, got %v", err)
	}
}

func TestResumeFromIdleErrors(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	if err := n.Resume(); !errors.Is(err, utils.ErrControl) {
		t.Errorf("Resume on idle must return ErrControl, got %v", err)
	}
}

func TestStopIdempotent(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	if err := n.Stop(); err != nil {
		t.Errorf("first Stop: %v", err)
	}
	if err := n.Stop(); err != nil {
		t.Errorf("second Stop: %v", err)
	}
}

// ============================================================================
// Coverage fillers — exercise the option mirrors that the convergence
// test does not happen to use, plus the rollback path on divergence.
// ============================================================================

func TestOptionFormConfigMethodsCoverage(t *testing.T) {
	t.Parallel()
	cfg := Config[float64]{}
	WithWeightInit[float64](WeightInitHe)(&cfg)
	WithBatchCallback[float64](func(uint, float64) {})(&cfg)
	if cfg.WeightInit != WeightInitHe {
		t.Errorf("WithWeightInit: got %v; want he", cfg.WeightInit)
	}
	if cfg.BatchCallback == nil {
		t.Errorf("WithBatchCallback did not register callback")
	}
}

func TestPresetRegressionMultiHiddenSurface(t *testing.T) {
	t.Parallel()
	cfg := Config[float64]{}
	PresetRegression[float64](8, 16)(&cfg)
	if cfg.InputSize != 8 {
		t.Errorf("InputSize = %d; want 8", cfg.InputSize)
	}
	if len(cfg.HiddenLayers) != 2 {
		t.Errorf("HiddenLayers count = %d; want 2", len(cfg.HiddenLayers))
	}
	if cfg.HiddenLayers[1].Size != 8 {
		t.Errorf("second hidden size = %d; want 8 (16/2)", cfg.HiddenLayers[1].Size)
	}
	if cfg.OutputActivation != activation.Linear {
		t.Errorf("OutputActivation = %v; want Linear", cfg.OutputActivation)
	}
}

func TestPresetRegressionHalfFloor(t *testing.T) {
	t.Parallel()
	cfg := Config[float64]{}
	PresetRegression[float64](2, 3)(&cfg)
	// 3/2 = 1, must floor to 2.
	if cfg.HiddenLayers[1].Size != 2 {
		t.Errorf("half-floor failed: size = %d; want 2", cfg.HiddenLayers[1].Size)
	}
}

func TestRestoreWeightsBringsBackSnapshot(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	original := n.snapshotWeights(nil)
	// Mutate every axon weight, then restore from snapshot.
	for _, hb := range n.Network.Hiddens {
		for _, h := range hb.Cells() {
			for i := range h.Axons {
				h.Axons[i].Weight = 99
			}
		}
	}
	for _, o := range n.Network.Output.Cells() {
		for i := range o.Axons {
			o.Axons[i].Weight = 99
		}
	}
	n.restoreWeights(original)
	after := n.snapshotWeights(nil)
	for i := range original {
		if original[i] != after[i] {
			t.Errorf("restoreWeights[%d]: got %v; want %v", i, after[i], original[i])
		}
	}
}

func TestFitDivergenceRollback(t *testing.T) {
	t.Parallel()
	// Configure a network with a wildly excessive learning rate so loss
	// increases epoch-over-epoch. With min-loss snapshot, Fit must
	// restore the best-observed weights and report the minimum loss.
	n := MustNew(
		PresetXOR[float64](),
		WithLearningRate[float64](50), // intentionally divergent
		WithMaxIterations[float64](10),
		WithLossLimit[float64](-1),
	)
	dataset := []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
	beforeFit := n.snapshotWeights(nil)
	_ = beforeFit // initial state — not directly compared below
	_, finalLoss, err := n.Fit(dataset)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// The recorded final loss is the minimum, not the divergent
	// late-epoch loss. We can't assert the exact value without re-running
	// the math, but we can assert it is finite (not NaN/Inf).
	if math.IsNaN(finalLoss) || math.IsInf(finalLoss, 0) {
		t.Errorf("post-rollback finalLoss is non-finite: %v", finalLoss)
	}
}

func TestQueryRejectsLengthMismatch(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	_, err := n.Query([]float64{1, 2, 3})
	if !errors.Is(err, utils.ErrInputData) {
		t.Errorf("Query length mismatch must return ErrInputData, got %v", err)
	}
}

func TestVerifyRejectsLengthMismatch(t *testing.T) {
	t.Parallel()
	n := MustNew(PresetXOR[float64]())
	_, err := n.Verify([]float64{1, 2}, []float64{1, 2})
	if !errors.Is(err, utils.ErrInputData) {
		t.Errorf("Verify target mismatch must return ErrInputData, got %v", err)
	}
}

func TestVerifyRejectsUncompiled(t *testing.T) {
	t.Parallel()
	n := NewBuilder[float64]()
	_, err := n.Verify([]float64{1, 2}, []float64{0})
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("Verify on uncompiled must return ErrUserConfig, got %v", err)
	}
}

func TestFitRejectsUncompiled(t *testing.T) {
	t.Parallel()
	n := NewBuilder[float64]()
	_, _, err := n.Fit([]Sample[float64]{{Input: []float64{0}, Target: []float64{0}}})
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("Fit on uncompiled must return ErrUserConfig, got %v", err)
	}
}

func TestStateString(t *testing.T) {
	t.Parallel()
	cases := []struct {
		s    state
		want string
	}{
		{stateUninitialized, "Uninitialized"},
		{stateConfiguring, "Configuring"},
		{stateOperational, "Operational"},
		{state(99), "unknown"},
	}
	for _, c := range cases {
		if got := c.s.String(); got != c.want {
			t.Errorf("state(%d).String() = %q; want %q", c.s, got, c.want)
		}
	}
}
