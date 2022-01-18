package perceptron

import (
	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
)

// NeuronBias.
func (nn *NN) NeuronBias() bool {
	return nn.Bias
}

// SetNeuronBias.
func (nn *NN) SetNeuronBias(bias bool) {
	nn.Bias = bias
}

// HiddenLayer.
func (nn *NN) HiddenLayer() []int {
	return params.CheckHiddenLayer(nn.Hidden)
}

// SetHiddenLayer.
func (nn *NN) SetHiddenLayer(layer ...int) {
	nn.Hidden = params.CheckHiddenLayer(layer)
}

// ActivationMode.
func (nn *NN) ActivationMode() uint8 {
	return nn.Activation
}

// SetActivationMode.
func (nn *NN) SetActivationMode(mode uint8) {
	nn.Activation = params.CheckActivationMode(mode)
}

// LossMode.
func (nn *NN) LossMode() uint8 {
	return nn.Loss
}

// SetLossMode.
func (nn *NN) SetLossMode(mode uint8) {
	nn.Loss = params.CheckLossMode(mode)
}

// LearningRate.
func (nn *NN) LearningRate() float64 {
	return float64(nn.Rate)
}

// SetLearningRate.
func (nn *NN) SetLearningRate(rate float64) {
	nn.Rate = params.CheckLearningRate(pkg.FloatType(rate))
}

// Weight.
func (nn *NN) Weight() pkg.Floater {
	return &nn.Weights
}

// SetWeight.
func (nn *NN) SetWeight(weight pkg.Floater) {
	if w, ok := weight.(pkg.Float3Type); ok {
		nn.Weights = w
	}
}
