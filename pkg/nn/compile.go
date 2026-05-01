package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/loss"
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
	// MVP: exactly one hidden layer required (validation enforces this).
	// Multi-hidden support is planned for v0.2 once pkg/network grows a
	// chain of Hidden bundles.
	hSpec := cfg.HiddenLayers[0]
	hidden := layer.NewDense[T](int(hSpec.Size), hSpec.Activation, hSpec.Bias)
	out := layer.NewOutput[T](int(cfg.OutputSize), cfg.OutputActivation, cfg.LossType, cfg.OutputBias)

	if err := n.SetLayers(in, hidden, out); err != nil {
		return utils.Wrap(utils.ErrUserConfig, err, "compile: SetLayers failed")
	}
	if err := n.Build(); err != nil {
		return utils.Wrap(utils.ErrUserConfig, err, "compile: Build failed")
	}
	n.LearningRate = cfg.LearningRate
	startProfilingServer(cfg.ProfilingAddr)
	return nil
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
			"compile: at least one Dense / WithHiddenLayer call is required (linear-only networks planned for v0.2)")
	}
	if len(cfg.HiddenLayers) > 1 {
		return utils.Newf(utils.ErrUserConfig,
			"compile: multi-hidden networks not supported in v0.1 (got %d hidden layers; planned for v0.2)",
			len(cfg.HiddenLayers))
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
