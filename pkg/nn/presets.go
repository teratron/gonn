package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// ============================================================================
// Higher-order options — pure compositions of the base WithX surface
// ============================================================================

// Sequential adds count hidden layers with identical size and activation.
// A one-liner replacement for repeated WithHiddenLayer calls.
//
// AI-Meta:
//   - Purpose: Add N identical hidden layers in one Options API call.
//   - Usage: nn.New[float32](Sequential[float32](3, 64, activation.ReLU), ...).
//   - Related: [Option], [WithHiddenLayer], [DeepNetwork].
//   - Stability: Stable.
func Sequential[T utils.Float](count, size uint, act activation.Type) Option[T] {
	return func(cfg *Config[T]) {
		for range count {
			cfg.HiddenLayers = append(cfg.HiddenLayers, HiddenLayerSpec[T]{
				Size:       size,
				Activation: act,
				Bias:       cfg.DefaultBias,
			})
		}
	}
}

// DeepNetwork adds layers hidden layers whose size halves at each level,
// floored at 2. Produces a funnel topology for progressive feature compression.
//
// AI-Meta:
//   - Purpose: Add a pyramid of hidden layers with halving size in the Options API.
//   - Usage: nn.New[float32](DeepNetwork[float32](128, 4, activation.ReLU), ...).
//   - Related: [Option], [Sequential], [WithHiddenLayer].
//   - Stability: Stable.
func DeepNetwork[T utils.Float](startSize, layers uint, act activation.Type) Option[T] {
	return func(cfg *Config[T]) {
		size := startSize
		for range layers {
			cfg.HiddenLayers = append(cfg.HiddenLayers, HiddenLayerSpec[T]{
				Size:       size,
				Activation: act,
				Bias:       cfg.DefaultBias,
			})
			size /= 2
			if size < 2 {
				size = 2
			}
		}
	}
}

// StandardSetup is an opinionated defaults bundle: given learning rate,
// MSE loss, bias on, Xavier init. Options applied later can override these.
//
// AI-Meta:
//   - Purpose: Apply a sensible baseline configuration in one Options API call.
//   - Usage: nn.New[float32](StandardSetup[float32](0.1), WithInput[float32](4), ...).
//   - Related: [Option], [WithLearningRate], [PresetXOR].
//   - Stability: Stable.
func StandardSetup[T utils.Float](rate T) Option[T] {
	return func(cfg *Config[T]) {
		cfg.LearningRate = rate
		cfg.LossType = loss.MSE
		cfg.DefaultBias = true
		cfg.WeightInit = WeightInitXavier
	}
}

// ============================================================================
// Presets — named option bundles for canonical tasks
// ============================================================================

// PresetXOR returns the canonical XOR network configuration:
// 2 → Sigmoid(4) → Sigmoid(1), MSE, rate 0.3, Xavier init, bias on.
//
// AI-Meta:
//   - Purpose: One-option XOR baseline for smoke tests and tutorials.
//   - Usage: n := nn.MustNew[float32](nn.PresetXOR[float32]()).
//   - Related: [Option], [MustNew], [StandardSetup].
//   - Stability: Stable.
func PresetXOR[T utils.Float]() Option[T] {
	return func(cfg *Config[T]) {
		cfg.InputSize = 2
		cfg.HiddenLayers = []HiddenLayerSpec[T]{
			{Size: 4, Activation: activation.SIGMOID, Bias: true},
		}
		cfg.OutputSize = 1
		cfg.OutputActivation = activation.SIGMOID
		cfg.OutputBias = true
		cfg.LossType = loss.MSE
		cfg.LearningRate = T(0.3)
		cfg.WeightInit = WeightInitXavier
		cfg.DefaultBias = true
	}
}

// PresetMNIST returns the canonical MNIST classifier:
// 784 → ReLU(128) → ReLU(64) → Softmax(10), CrossEntropy, He init.
//
// AI-Meta:
//   - Purpose: Standard multi-layer MNIST digit classifier preset.
//   - Usage: n, err := nn.New[float32](nn.PresetMNIST[float32]()).
//   - Related: [Option], [New], [PresetRegression].
//   - Stability: Stable.
func PresetMNIST[T utils.Float]() Option[T] {
	return func(cfg *Config[T]) {
		cfg.InputSize = 784
		cfg.HiddenLayers = []HiddenLayerSpec[T]{
			{Size: 128, Activation: activation.ReLU, Bias: true},
			{Size: 64, Activation: activation.ReLU, Bias: true},
		}
		cfg.OutputSize = 10
		cfg.OutputActivation = activation.SOFTMAX
		cfg.OutputBias = true
		cfg.LossType = loss.CCE
		cfg.WeightInit = WeightInitHe
		cfg.DefaultBias = true
	}
}

// PresetRegression returns a generic regression configuration:
// inputSize → ReLU(hiddenSize) → ReLU(hiddenSize/2) → Linear(1), MSE.
//
// AI-Meta:
//   - Purpose: Generic continuous-output regression preset parameterised by input and hidden sizes.
//   - Usage: n, err := nn.New[float32](nn.PresetRegression[float32](10, 64)).
//   - Related: [Option], [New], [PresetMNIST].
//   - Stability: Stable.
func PresetRegression[T utils.Float](inputSize, hiddenSize uint) Option[T] {
	half := max(hiddenSize/2, 2)
	return func(cfg *Config[T]) {
		cfg.InputSize = inputSize
		cfg.HiddenLayers = []HiddenLayerSpec[T]{
			{Size: hiddenSize, Activation: activation.ReLU, Bias: true},
			{Size: half, Activation: activation.ReLU, Bias: true},
		}
		cfg.OutputSize = 1
		cfg.OutputActivation = activation.Linear
		cfg.OutputBias = true
		cfg.LossType = loss.MSE
		cfg.WeightInit = WeightInitHe
		cfg.DefaultBias = true
	}
}
