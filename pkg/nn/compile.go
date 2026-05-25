package nn

import (
	"errors"
	"fmt"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/layer/conv"
	normPkg "github.com/teratron/gonn/pkg/layer/norm"
	"github.com/teratron/gonn/pkg/layer/recurrent"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/utils"
	"github.com/teratron/gonn/pkg/visualization"
)

// compile is the shared finalisation routine consumed by both the
// Builder API ([Compile]) and the Functional Options API ([New]).
// Per [l2-nn-facade] §5.4 there is exactly one compile path — both
// fluent styles end here.
//
// The function:
//  1. Applies defaults (LearningRate, LossLimit, MaxIterations, WeightInit).
//  2. Validates the staged configuration against [l2-nn-facade] §5.7.
//  3. Builds the underlying network.Network[T] via Phase-1 constructors.
//
// Every returned error wraps utils.ErrUserConfig per the project
// taxonomy (C32). Soft warnings emit Logger.Warn but do not block.
func compile[T utils.Float](n *NN[T], cfg *Config[T]) error {
	cfg.applyDefaults()

	if err := validate(cfg); err != nil {
		return err
	}
	emitSoftWarnings(cfg)

	// Resolve the conv prefix BEFORE building Input — when present, the
	// Input layer is sized to the conv stack's final output length, not
	// the raw input length the user declared via WithInput. The raw size
	// is preserved on NN for SetInputs-style validation in Train / Query.
	rawInputSize := cfg.InputSize
	effectiveInputSize := cfg.InputSize
	if len(cfg.ConvPrefix) > 0 {
		// Pre-pass: validate recurrent layer shapes before the 2-D conv and
		// dummy-Forward walks. This catches seqLen/inSize/hidden mismatches
		// with an actionable error message before computeConvChainOutput runs.
		if hasRecurrentLayer(cfg.ConvPrefix) {
			if err := setupRecurrentShapes(cfg); err != nil {
				return utils.Wrap(utils.ErrUserConfig, err, "compile: recurrent shape propagation")
			}
		}
		// Pre-pass: propagate (C, H, W) through any Conv2D/MaxPool2D/AvgPool2D/
		// Flatten2D layers so each layer's OutputShape() resolves during the
		// dummy-Forward shape walk below.
		if err := setupConv2DShapes(cfg); err != nil {
			return utils.Wrap(utils.ErrUserConfig, err, "compile: conv2d shape propagation")
		}
		convOut, err := computeConvChainOutput(cfg.ConvPrefix, int(rawInputSize))
		if err != nil {
			return utils.Wrap(utils.ErrUserConfig, err, "compile: conv prefix shape resolution")
		}
		if convOut <= 0 {
			return utils.Newf(utils.ErrUserConfig,
				"compile: conv prefix collapses input length %d to zero — check kernel/pool sizes",
				rawInputSize)
		}
		effectiveInputSize = uint(convOut)
	}
	n.rawInputSize = rawInputSize
	n.convPrefix = cfg.ConvPrefix

	in := layer.NewInput[T](int(effectiveInputSize))
	// Build the multi-hidden chain per [l2-multihidden-impl] §5.5. Each
	// HiddenLayerSpec carries its own Size / Activation / Bias, so the
	// chain composes mixed-activation, mixed-bias topologies in one pass.
	// validate() above enforces Size > 0 and known Activation per entry.
	hiddens := make([]*layer.Dense[T], len(cfg.HiddenLayers))
	for i, hSpec := range cfg.HiddenLayers {
		hiddens[i] = layer.NewDense[T](int(hSpec.Size), hSpec.Activation, hSpec.Bias)
	}
	out := layer.NewOutput[T](int(cfg.OutputSize), cfg.OutputActivation, cfg.LossType, cfg.OutputBias)

	if err := n.SetLayers(in, hiddens, out); err != nil {
		return utils.Wrap(utils.ErrUserConfig, err, "compile: SetLayers failed")
	}

	// Wire the weight-init sampler before Build so axons receive the correct
	// initial values (fixes the known axon.New U[-0.5,0.5] debt — T-6B06).
	rng, _ := utils.NewRNG(cfg.WeightInitSeed)
	n.SetWeightSampler(weightSamplerFor[T](cfg.WeightInit, rng))

	if err := n.Build(); err != nil {
		return utils.Wrap(utils.ErrUserConfig, err, "compile: Build failed")
	}
	// Initialise conv-stack kernel weights via the same RNG so reproducibility
	// (WI-2 seed contract) extends to convolutional kernels. Only Conv1D
	// layers carry trainable weights; Pool / Flatten implement Init as a
	// type assertion no-op.
	for _, cl := range n.convPrefix {
		if init, ok := cl.(interface{ Init(rng *rand.Rand) }); ok {
			init.Init(rng)
		}
	}
	n.LearningRate = cfg.LearningRate

	// Resolve optimizer: use the user-supplied instance or fall back to SGD.
	if cfg.Optimizer != nil {
		n.opt = cfg.Optimizer
	} else {
		n.opt = optimizer.DefaultOptimizer(cfg.LearningRate)
	}
	n.reg = cfg.Regularizer
	n.sched = cfg.Scheduler
	n.SetTopologyMode(cfg.TopologyMode)

	// Resolve nil norm-layer entries inserted by WithBatchNorm/WithLayerNorm
	// when the hidden layer sizes were not yet declared at option-apply time.
	if len(cfg.NormLayers) > 0 {
		resolved := make(map[int]normPkg.Normalizer[T], len(cfg.NormLayers))
		for idx, nl := range cfg.NormLayers {
			if nl != nil {
				resolved[idx] = nl
				continue
			}
			// nil sentinel: create a default BatchNorm using the hidden layer size.
			if idx < len(cfg.HiddenLayers) {
				size := int(cfg.HiddenLayers[idx].Size)
				if size > 0 {
					resolved[idx] = normPkg.NewBatchNorm[T](size)
				}
			}
		}
		n.normLayers = resolved
	}
	n.callbacks = cfg.Callbacks

	// Validate MetaLearner: reject if the outer network is currently training
	// (META-state guard per l2-meta-learning-impl §5.4, ErrMetaLearnerRunning).
	if cfg.MetaLearner != nil && n.control.Load() != controlIdle {
		return utils.Newf(utils.ErrMetaLearnerRunning,
			"compile: WithMetaLearner cannot be applied while the network is training")
	}

	// Wire compute backend. Probe the requested backend via Allocate(1);
	// on ErrBackendUnavailable fall back to the CPU reference (COMP-3).
	n.backend = resolveBackend(cfg.Backend)

	startProfilingServer(cfg.ProfilingAddr)
	if err := startVisServer(n, cfg); err != nil {
		return err
	}
	return nil
}

// resolveBackend returns a ready Backend[T]. When b is nil or returns
// ErrBackendUnavailable on a 1-element Allocate probe, the always-available
// CPU backend is substituted and a Warn is emitted (l1-compute-backend §5.3).
func resolveBackend[T utils.Float](b compute.Backend[T]) compute.Backend[T] {
	if b != nil {
		if _, err := b.Allocate(1); !errors.Is(err, utils.ErrBackendUnavailable) {
			// Backend is live — use it.
			return b
		}
		utils.Logger.Warn("backend unavailable, falling back to CPU",
			"backend", b.Name(), "err", utils.ErrBackendUnavailable)
	}
	// CPU backend is always registered via the blank import in init.go.
	cpu, _ := compute.Get[T]("cpu")
	return cpu
}

// startVisServer starts the visualization HTTP server when cfg.VisAddr is
// non-empty and registers a snapshot callback that pulls current state from n.
func startVisServer[T utils.Float](n *NN[T], cfg *Config[T]) error {
	if cfg.VisAddr == "" {
		return nil
	}
	vs := visualization.NewVisServer(cfg.VisAddr, cfg.VisToken, cfg.VisCORS)
	vs.RegisterNetwork(func() visualization.NetworkState {
		return visualization.NetworkState{
			TopologyVersion: n.Network.TopologyVersion(),
			Control:         "idle",
		}
	})
	if err := vs.Start(); err != nil {
		return utils.Wrap(utils.ErrIO, err, "compile: visualization server start failed on %q", cfg.VisAddr)
	}
	n.vis = vs
	return nil
}

// setupConv2DShapes walks the conv prefix and propagates the (C, H, W)
// shape declared via [WithInputShape] (or auto-inferred for single-channel
// square inputs) through every Conv2D / MaxPool2D / AvgPool2D / Flatten2D
// layer. Each layer's SetInputShape is called so subsequent OutputShape()
// queries succeed without requiring a Forward pass first.
//
// 1-D conv layers (Conv1D / MaxPool1D / AvgPool1D / Flatten) interrupt the
// 2-D chain: their flat output length is the only contract, so after a 1-D
// layer the (C, H, W) tracker is cleared. Mixed chains are unusual but the
// function handles them deterministically.
//
// AI-Meta:
//   - Purpose: Pre-pass that propagates CHW shape through 2-D conv layers before computeConvChainOutput.
//   - Concurrency: Safe; mutates layer InH/InW/InChannels fields, not shared state.
//   - Related: [compile], [computeConvChainOutput], [conv.Conv2D.SetInputShape].
func setupConv2DShapes[T utils.Float](cfg *Config[T]) error {
	curC, curH, curW := cfg.InputC, cfg.InputH, cfg.InputW
	// Auto-infer single-channel square shape from the raw input length when
	// the user did not declare an explicit shape (MNIST 784 → (1, 28, 28)).
	if curC == 0 && curH == 0 && curW == 0 {
		raw := int(cfg.InputSize)
		s := iSqrt(raw)
		if s > 0 && s*s == raw {
			curC, curH, curW = 1, s, s
		}
	}
	if curC == 0 || curH == 0 || curW == 0 {
		// No 2-D shape available — only 1-D layers (or no layers) supported.
		// Validate that no 2-D layer is present without shape declaration.
		for i, cl := range cfg.ConvPrefix {
			switch cl.(type) {
			case *conv.Conv2D[T], *conv.MaxPool2D[T], *conv.AvgPool2D[T], *conv.Flatten2D[T]:
				return fmt.Errorf("conv layer %d is 2-D but no input shape declared (use WithInputShape(c, h, w)): %w",
					i, utils.ErrConv2DShapeMismatch)
			}
		}
		return nil
	}

	if int(cfg.InputSize) != curC*curH*curW {
		return fmt.Errorf("InputSize %d != InputC*InputH*InputW (%d*%d*%d = %d): %w",
			cfg.InputSize, curC, curH, curW, curC*curH*curW, utils.ErrConv2DShapeMismatch)
	}

	in2D := true
	for i, cl := range cfg.ConvPrefix {
		switch l := cl.(type) {
		case *conv.Conv2D[T]:
			if !in2D {
				return fmt.Errorf("conv layer %d is Conv2D but follows a 1-D layer: %w",
					i, utils.ErrConv2DShapeMismatch)
			}
			if l.InChannels != curC {
				return fmt.Errorf("Conv2D layer %d declares InChannels=%d but receives %d channels: %w",
					i, l.InChannels, curC, utils.ErrConv2DShapeMismatch)
			}
			l.SetInputShape(curH, curW)
			if err := l.Validate(curC, curH, curW); err != nil {
				return fmt.Errorf("Conv2D layer %d validate: %w", i, err)
			}
			oh, ow := l.OutputShape()
			if oh == 0 || ow == 0 {
				return fmt.Errorf("Conv2D layer %d collapses shape (%d, %d, %d) to zero: %w",
					i, curC, curH, curW, utils.ErrConv2DShapeMismatch)
			}
			curC, curH, curW = l.NumFilters, oh, ow
		case *conv.MaxPool2D[T]:
			if !in2D {
				return fmt.Errorf("conv layer %d is MaxPool2D but follows a 1-D layer: %w",
					i, utils.ErrConv2DShapeMismatch)
			}
			l.SetInputShape(curC, curH, curW)
			if err := l.Validate(curC, curH, curW); err != nil {
				return fmt.Errorf("MaxPool2D layer %d validate: %w", i, err)
			}
			oh, ow := l.OutputShape()
			curH, curW = oh, ow
		case *conv.AvgPool2D[T]:
			if !in2D {
				return fmt.Errorf("conv layer %d is AvgPool2D but follows a 1-D layer: %w",
					i, utils.ErrConv2DShapeMismatch)
			}
			l.SetInputShape(curC, curH, curW)
			if err := l.Validate(curC, curH, curW); err != nil {
				return fmt.Errorf("AvgPool2D layer %d validate: %w", i, err)
			}
			oh, ow := l.OutputShape()
			curH, curW = oh, ow
		case *conv.Flatten2D[T]:
			if !in2D {
				return fmt.Errorf("conv layer %d is Flatten2D but follows a 1-D layer: %w",
					i, utils.ErrConv2DShapeMismatch)
			}
			l.SetInputShape(curC, curH, curW)
			in2D = false // After Flatten2D the chain emits a flat vector — Conv1D-style layers may follow.
		default:
			// 1-D layer (Conv1D / MaxPool1D / AvgPool1D / Flatten) — the 2-D
			// tracker is no longer meaningful. Subsequent 2-D layers are an
			// error caught by the in2D guard.
			in2D = false
		}
	}
	return nil
}

// hasRecurrentLayer reports whether chain contains any recurrent layer
// (GRU, LSTM, SimpleRNN, or LastStep).
func hasRecurrentLayer[T utils.Float](chain []conv.Layer[T]) bool {
	for _, cl := range chain {
		switch cl.(type) {
		case *recurrent.GRU[T], *recurrent.LSTM[T], *recurrent.SimpleRNN[T], *recurrent.LastStep[T]:
			return true
		}
	}
	return false
}

// setupRecurrentShapes validates that recurrent layers in cfg.ConvPrefix have
// self-consistent shapes and that each layer's InputSize() matches the
// preceding layer's OutputSize() (or the raw cfg.InputSize for the first
// layer). Non-recurrent layers interrupt tracking — their output is unknown
// without a Forward pass, which is left to computeConvChainOutput.
//
// AI-Meta:
//   - Purpose: Pre-pass shape validation for recurrent prefix layers; provides actionable errors before computeConvChainOutput.
//   - Concurrency: Safe; read-only on layer fields.
//   - Related: [compile], [computeConvChainOutput], [hasRecurrentLayer].
func setupRecurrentShapes[T utils.Float](cfg *Config[T]) error {
	cur := int(cfg.InputSize)
	for i, cl := range cfg.ConvPrefix {
		switch l := cl.(type) {
		case *recurrent.GRU[T]:
			if l.SeqLen <= 0 || l.InSize <= 0 || l.Hidden <= 0 {
				return fmt.Errorf("GRU layer %d: seqLen=%d inSize=%d hidden=%d must all be >0: %w",
					i, l.SeqLen, l.InSize, l.Hidden, utils.ErrRecurrentShapeMismatch)
			}
			if cur > 0 && cur != l.InputSize() {
				return fmt.Errorf("GRU layer %d expects input %d (seqLen=%d * inSize=%d) but receives %d: %w",
					i, l.InputSize(), l.SeqLen, l.InSize, cur, utils.ErrRecurrentShapeMismatch)
			}
			cur = l.OutputSize()
		case *recurrent.LSTM[T]:
			if l.SeqLen <= 0 || l.InSize <= 0 || l.Hidden <= 0 {
				return fmt.Errorf("LSTM layer %d: seqLen=%d inSize=%d hidden=%d must all be >0: %w",
					i, l.SeqLen, l.InSize, l.Hidden, utils.ErrRecurrentShapeMismatch)
			}
			if cur > 0 && cur != l.InputSize() {
				return fmt.Errorf("LSTM layer %d expects input %d (seqLen=%d * inSize=%d) but receives %d: %w",
					i, l.InputSize(), l.SeqLen, l.InSize, cur, utils.ErrRecurrentShapeMismatch)
			}
			cur = l.OutputSize()
		case *recurrent.SimpleRNN[T]:
			if l.SeqLen <= 0 || l.InSize <= 0 || l.Hidden <= 0 {
				return fmt.Errorf("SimpleRNN layer %d: seqLen=%d inSize=%d hidden=%d must all be >0: %w",
					i, l.SeqLen, l.InSize, l.Hidden, utils.ErrRecurrentShapeMismatch)
			}
			if cur > 0 && cur != l.InputSize() {
				return fmt.Errorf("SimpleRNN layer %d expects input %d (seqLen=%d * inSize=%d) but receives %d: %w",
					i, l.InputSize(), l.SeqLen, l.InSize, cur, utils.ErrRecurrentShapeMismatch)
			}
			cur = l.OutputSize()
		case *recurrent.LastStep[T]:
			if l.SeqLen <= 0 || l.Hidden <= 0 {
				return fmt.Errorf("LastStep layer %d: seqLen=%d hidden=%d must all be >0: %w",
					i, l.SeqLen, l.Hidden, utils.ErrRecurrentShapeMismatch)
			}
			if cur > 0 && cur != l.InputSize() {
				return fmt.Errorf("LastStep layer %d expects input %d (seqLen=%d * hidden=%d) but receives %d: %w",
					i, l.InputSize(), l.SeqLen, l.Hidden, cur, utils.ErrRecurrentShapeMismatch)
			}
			cur = l.OutputSize()
		default:
			// Non-recurrent layer: output size unknown before Forward.
			// Reset tracker; computeConvChainOutput handles shape from here.
			cur = 0
		}
	}
	return nil
}

// iSqrt is the package-level mirror of conv.isqrt (avoids importing private
// helpers across packages). Returns the integer square root of n; zero for
// n ≤ 0.
func iSqrt(n int) int {
	if n <= 0 {
		return 0
	}
	x := n
	y := (x + 1) / 2
	for y < x {
		x = y
		y = (x + n/x) / 2
	}
	return x
}

// computeConvChainOutput walks the conv prefix once with a dummy input of
// length rawInputSize and returns the chain's final output length. Each
// layer's Forward is invoked on a zero-filled scratch buffer purely to let
// the layer record its inLen; the returned length comes from
// [conv.Layer.OutputSize]. Layers whose OutputSize is zero on a non-zero
// input indicate a shape mismatch (e.g. kernel larger than the feature
// map) and the function returns an error wrapping
// [utils.ErrConvShapeMismatch].
//
// AI-Meta:
//   - Purpose: Shape-resolve the conv prefix at compile time so the Input layer gets the right size.
//   - Concurrency: Safe; each layer is exercised once with a freshly allocated zero buffer.
//   - Related: [compile], [conv.Layer], [conv.NewConv1D].
func computeConvChainOutput[T utils.Float](chain []conv.Layer[T], rawInputSize int) (int, error) {
	cur := rawInputSize
	for i, cl := range chain {
		if cur <= 0 {
			return 0, fmt.Errorf("conv layer %d sees zero-length input: %w",
				i, utils.ErrConvShapeMismatch)
		}
		out := cl.Forward(make([]T, cur))
		size := len(out)
		if size == 0 {
			return 0, fmt.Errorf("conv layer %d produced empty output from input length %d: %w",
				i, cur, utils.ErrConvShapeMismatch)
		}
		cur = size
	}
	return cur, nil
}

// weightSamplerFor converts a WeightInitMethod into the sampler function
// consumed by network.Network.SetWeightSampler. Xavier is the default when
// method is unrecognised (should not happen after validate()).
func weightSamplerFor[T utils.Float](method WeightInitMethod, rng *rand.Rand) network.WeightSampler[T] {
	switch method {
	case WeightInitHe:
		return func(fanIn, _ int) T { return utils.HeNormal[T](rng, fanIn) }
	case WeightInitRandom:
		return func(_, _ int) T { return utils.Uniform[T](rng) }
	default: // WeightInitXavier
		return func(fanIn, fanOut int) T { return utils.XavierUniform[T](rng, fanIn, fanOut) }
	}
}

// validate enforces the [l2-nn-facade] §5.7 hard-error rules. The check
// order matches the spec's table top-to-bottom — predictable for users
// who diff Compile() error messages between runs.
func validate[T utils.Float](cfg *Config[T]) error {
	if cfg.InputSize == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"compile: Input layer is missing or has size 0 (call Input(size) / WithInput(size) with size > 0)")
	}
	if cfg.OutputSize == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"compile: Output layer is missing or has size 0 (call Output(size, ...) / WithOutput(size, ...) with size > 0)")
	}
	if len(cfg.HiddenLayers) == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"compile: at least one Dense / WithHiddenLayer call is required (linear-only networks planned for v0.6)")
	}
	for i, h := range cfg.HiddenLayers {
		if h.Size == 0 {
			return utils.Newf(utils.ErrUserConfig,
				"compile: hidden layer %d has size 0 — every Dense / WithHiddenLayer must declare size > 0", i)
		}
		if !isKnownActivation(h.Activation) {
			return utils.Newf(utils.ErrUserConfig,
				"compile: hidden layer %d uses unregistered activation %d", i, uint8(h.Activation))
		}
	}
	if cfg.LearningRate <= 0 {
		return utils.Newf(utils.ErrUserConfig,
			"compile: learning rate must be positive, got %v (use WithLearningRate(rate))", cfg.LearningRate)
	}
	if cfg.MaxIterations == 0 {
		return utils.Newf(utils.ErrUserConfig,
			"compile: max iterations must be positive (use WithMaxIterations(count))")
	}
	if !isKnownActivation(cfg.OutputActivation) {
		return utils.Newf(utils.ErrUserConfig,
			"compile: output activation %d is not registered in the activation dispatcher", uint8(cfg.OutputActivation))
	}
	if !isKnownLoss(cfg.LossType) {
		return utils.Newf(utils.ErrUserConfig,
			"compile: loss symbol %d is not registered in the loss dispatcher", uint8(cfg.LossType))
	}
	if !isKnownWeightInit(cfg.WeightInit) {
		return utils.Newf(utils.ErrUserConfig,
			"compile: weight-init method %q is not in {xavier, he, random}", string(cfg.WeightInit))
	}
	return nil
}

// emitSoftWarnings logs the [l2-nn-facade] §5.7 advisory checks. These
// do not block compilation — they catch common misconfigurations that
// would still produce a runnable but suboptimal network.
func emitSoftWarnings[T utils.Float](cfg *Config[T]) {
	if cfg.OutputActivation == activation.SOFTMAX && cfg.LossType == loss.MSE {
		utils.Logger.Warn(
			"output activation SOFTMAX paired with loss MSE — CrossEntropy variants typically fit better",
		)
	}
	if cfg.OutputActivation == activation.SIGMOID && cfg.LossType == loss.CCE && cfg.OutputSize > 1 {
		utils.Logger.Warn(
			"output activation SIGMOID paired with CrossEntropy and OutputSize > 1 — consider BCE for binary or SOFTMAX+CrossEntropy for multi-class",
			"outputSize", cfg.OutputSize,
		)
	}
	if len(cfg.HiddenLayers) > 5 && cfg.WeightInit == WeightInitRandom {
		utils.Logger.Warn(
			"deep stack (>5 hidden) with WeightInitRandom — gradient explosion risk; prefer Xavier or He",
			"hiddenCount", len(cfg.HiddenLayers),
		)
	}
	if len(cfg.HiddenLayers) > DefaultExcessiveLayersWarning {
		utils.Logger.Warn(
			"layer count exceeds 10,000 — construction is O(N) but training cost scales with topology",
			"hiddenCount", len(cfg.HiddenLayers),
		)
	}
}

// isKnownActivation returns true when t is one of the symbols defined
// in the activation dispatcher. The check is the cheapest way to catch
// "user typed a wrong cast" without re-running the full Activation()
// switch.
func isKnownActivation(t activation.Type) bool {
	switch t {
	case activation.ELISH, activation.ELU, activation.Linear,
		activation.LeakyReLU, activation.ReLU, activation.SELU,
		activation.SIGMOID, activation.SOFTMAX, activation.SWISH,
		activation.TanH:
		return true
	}
	return false
}

// isKnownLoss mirrors isKnownActivation for the loss dispatcher.
func isKnownLoss(l loss.Type) bool {
	switch l {
	case loss.MSE, loss.MAE, loss.CCE, loss.BCE, loss.CROSS_ENTROPY,
		loss.MAPE, loss.MSLE, loss.KLD, loss.COSINE, loss.POISSON,
		loss.HINGE, loss.SQ_HINGE, loss.CAT_HINGE, loss.LOG_COSH,
		loss.HUBER, loss.AVG, loss.RMSE, loss.ARCTAN:
		return true
	}
	return false
}

// isKnownWeightInit reports whether m is one of the strategies defined
// in §5.6.
func isKnownWeightInit(m WeightInitMethod) bool {
	switch m {
	case WeightInitXavier, WeightInitHe, WeightInitRandom:
		return true
	}
	return false
}
