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
// Useful as a one-liner replacement for repeated WithHiddenLayer calls.
//
// Note: in v0.1 the network only supports a single hidden layer. Calling
// Sequential with count > 1 is accepted by the staging buffer and rejected
// by Compile() — this is intentional so the full configuration surfaces
// in error messages.
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

// DeepNetwork adds layers hidden levels of progressively halving size,
// floored at 2. Replicates the convenience helper from the v3 reference
// design. Same v0.1 multi-hidden caveat as [Sequential].
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

// StandardSetup is an opinionated defaults bundle: learning rate,
// MSE loss, bias on, Xavier init. Application order means callers who
// supply WithLearningRate later override the rate set here.
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

// PresetXOR returns the canonical XOR-network configuration:
// 2 → Sigmoid(4) → Sigmoid(1), MSE, rate 0.3, Xavier init, bias on.
//
// Validates against the Phase-1 XOR smoke test so a regression in this
// preset shows up as a Phase-2 test failure.
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
// 784 → ReLU(128) → ReLU(64) → SoftMax(10), CrossEntropy, He init.
//
// Multi-hidden — runs into the v0.1 single-hidden Compile() check.
// Provided so the preset surface matches the spec; v0.2 will lift the
// restriction and this preset will work without modification.
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

// PresetRegression returns a generic regression configuration with the
// supplied input and hidden size:
// inputSize → ReLU(hiddenSize) → ReLU(hiddenSize/2) → Linear(1), MSE.
//
// Multi-hidden — same v0.1 limitation as [PresetMNIST]; ships against
// the spec for forward compatibility.
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
