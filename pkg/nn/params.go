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
func (nn *NN[T]) SetBias(bias bool) {
	nn.Bias = bias
}

// GetHiddenLayers.
func (nn *NN[T]) GetHiddenLayers() []uint {
	return checkLayers(nn.HiddenLayers)
}

// SetHiddenLayers.
func (nn *NN[T]) SetHiddenLayers(layers ...uint) {
	nn.HiddenLayers = checkLayers(layers)
}

func checkLayers(layers []uint) []uint {
	if layers != nil && len(layers) > 0 {
		return layers
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
func (nn *NN[T]) SetLossLimit(limit T) {
	nn.LossLimit = limit
}

// GetRate.
func (nn *NN[T]) GetRate() T {
	return nn.Rate
}

// SetRate.
func (nn *NN[T]) SetRate(r T) {
	nn.Rate = rate.CheckLearningRate(r)
}

// GetWeights.
func (nn *NN[T]) GetWeights() [][][]T {
	return &nn.Weights
}

// SetWeights.
func (nn *NN[T]) SetWeights(weight T) {
	if w, ok := weight.([][][]T); ok {
		nn.Weights = w
	}
}

// GetLengthInput.
func (nn *NN[T]) GetLengthInput() int {
	return nn.lenInput
}

// GetLengthOutput.
func (nn *NN[T]) GetLengthOutput() int {
	return nn.lenOutput
}
