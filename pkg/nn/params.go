package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/rate"
)

// GetBias.
func (nn *NN[T]) GetBias() bool {
	return nn.Bias
}

// SetBias.
func (nn *NN[T]) SetBias(state bool) {
	nn.Bias = state
}

// GetHiddenLayers.
func (nn *NN[T]) GetHiddenLayers() []uint {
	return checkLayers(nn.HiddenLayers)
}

// SetHiddenLayers.
func (nn *NN[T]) SetHiddenLayers(data ...uint) {
	nn.HiddenLayers = checkLayers(data)
}

func checkLayers(data []uint) []uint {
	if data != nil && len(data) > 0 {
		return data
	}

	return []uint{0}
}

// GetActivationMode.
func (nn *NN[T]) GetActivationMode() activation.Type {
	return nn.ActivationMode
}

// SetActivationMode.
func (nn *NN[T]) SetActivationMode(mode activation.Type) {
	nn.ActivationMode = activation.CheckActivationMode(mode)
}

// GetLossMode.
func (nn *NN[T]) GetLossMode() loss.Type {
	return nn.LossMode
}

// SetLossMode.
func (nn *NN[T]) SetLossMode(mode loss.Type) {
	nn.LossMode = loss.CheckLossMode(mode)
}

// GetLossLimit.
func (nn *NN[T]) GetLossLimit() T {
	return nn.LossLimit
}

// SetLossLimit.
func (nn *NN[T]) SetLossLimit(value T) {
	nn.LossLimit = value
}

// GetRate.
func (nn *NN[T]) GetRate() T {
	return nn.Rate
}

// SetRate.
func (nn *NN[T]) SetRate(value T) {
	nn.Rate = rate.CheckLearningRate(value)
}

// GetWeights.
func (nn *NN[T]) GetWeights() *[][][]T {
	return &nn.Weights
}

// SetWeights.
func (nn *NN[T]) SetWeights(data [][][]T) {
	nn.Weights = data // TODO:
}

// GetLengthInput.
func (nn *NN[T]) GetLengthInput() int {
	return nn.lenInput
}

// GetLengthOutput.
func (nn *NN[T]) GetLengthOutput() int {
	return nn.lenOutput
}
