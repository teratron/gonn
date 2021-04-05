package perceptron

import (
	"github.com/teratron/gonn"
	"github.com/teratron/gonn/params"
)

// NeuronBias
func (nn *NN) NeuronBias() bool {
	return nn.Bias
}

// SetNeuronBias
func (nn *NN) SetNeuronBias(bias bool) {
	nn.Bias = bias
}

// HiddenLayer
func (nn *NN) HiddenLayer() []int {
	return params.CheckHiddenLayer(nn.Hidden)
}

// SetHiddenLayer
func (nn *NN) SetHiddenLayer(layer ...int) {
	nn.Hidden = params.CheckHiddenLayer(layer)
}

// ActivationMode
func (nn *NN) ActivationMode() uint8 {
	return nn.Activation
}

// SetActivationMode
func (nn *NN) SetActivationMode(mode uint8) {
	nn.Activation = params.CheckActivationMode(mode)
}

// LossMode
func (nn *NN) LossMode() uint8 {
	return nn.Loss
}

// SetLossMode
func (nn *NN) SetLossMode(mode uint8) {
	nn.Loss = params.CheckLossMode(mode)
}

// LossLimit
func (nn *NN) LossLimit() float64 {
	return nn.Limit
}

// SetLossLimit
func (nn *NN) SetLossLimit(limit float64) {
	nn.Limit = limit
}

// LearningRate
func (nn *NN) LearningRate() float64 {
	return nn.Rate
}

// SetLearningRate
func (nn *NN) SetLearningRate(rate float64) {
	nn.Rate = params.CheckLearningRate(rate)
}

// Weight
func (nn *NN) Weight() gonn.Floater {
	return &nn.Weights
}

// SetWeight
func (nn *NN) SetWeight(weight gonn.Floater) {
	if w, ok := weight.(gonn.Float3Type); ok {
		nn.Weights = w
	}
}
