package nn

import (
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/utils"
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

	in := layer.NewInput[T](int(cfg.InputSize))
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
	rng, _ := utils.NewRNG(0)
	n.Network.SetWeightSampler(weightSamplerFor[T](cfg.WeightInit, rng))

	if err := n.Build(); err != nil {
		return utils.Wrap(utils.ErrUserConfig, err, "compile: Build failed")
	}
	n.LearningRate = cfg.LearningRate

	// Resolve optimizer: use the user-supplied instance or fall back to SGD.
	if cfg.Optimizer != nil {
		n.opt = cfg.Optimizer
	} else {
		n.opt = optimizer.DefaultOptimizer[T](cfg.LearningRate)
	}
	n.reg = cfg.Regularizer

	startProfilingServer(cfg.ProfilingAddr)
	return nil
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
