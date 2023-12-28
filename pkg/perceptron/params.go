package perceptron

import (
	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layers"
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

// GetHiddenLayer.
func (nn *NN[T]) GetHiddenLayer() []uint {
	return layers.CheckLayers(nn.HiddenLayer)
}

// SetHiddenLayer.
func (nn *NN[T]) SetHiddenLayer(layer ...uint) {
	nn.HiddenLayer = layers.CheckLayers(layer)
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
func (nn *NN[T]) SetWeights(weight pkg.Floater) {
	if w, ok := weight.(nn.Float3Type); ok {
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
