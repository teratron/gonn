package perceptron

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/params"
)

// GetBias.
func (nn *NN) GetBias() bool {
	return nn.Bias
}

// SetBias.
func (nn *NN) SetBias(bias bool) {
	nn.Bias = bias
}

// GetHiddenLayer.
func (nn *NN) GetHiddenLayer() []uint {
	return params.CheckLayer(nn.HiddenLayer)
}

// SetHiddenLayer.
func (nn *NN) SetHiddenLayer(layer ...uint) {
	nn.HiddenLayer = params.CheckLayer(layer)
}

// GetActivationMode.
func (nn *NN) GetActivationMode() uint8 {
	return nn.ActivationMode
}

// SetActivationMode.
func (nn *NN) SetActivationMode(mode uint8) {
	nn.ActivationMode = activation.CheckActivationMode(mode)
}

// GetLossMode.
func (nn *NN) GetLossMode() uint8 {
	return nn.LossMode
}

// SetLossMode.
func (nn *NN) SetLossMode(mode uint8) {
	nn.LossMode = loss.CheckLossMode(mode)
}

// GetLossLimit.
func (nn *NN) GetLossLimit() float64 {
	return nn.LossLimit
}

// SetLossLimit.
func (nn *NN) SetLossLimit(limit float64) {
	nn.LossLimit = limit
}

// GetRate.
func (nn *NN) GetRate() float64 {
	return float64(nn.Rate)
}

// SetRate.
func (nn *NN) SetRate(rate float64) {
	nn.Rate = rate.CheckLearningRate(nn.FloatType(rate))
}

// GetWeights.
func (nn *NN) GetWeights() nn.Floater {
	return &nn.Weights
}

// SetWeights.
func (nn *NN) SetWeights(weight nn.Floater) {
	if w, ok := weight.(nn.Float3Type); ok {
		nn.Weights = w
	}
}

// GetLengthInput.
func (nn *NN) GetLengthInput() int {
	return nn.lenInput
}

// GetLengthOutput.
func (nn *NN) GetLengthOutput() int {
	return nn.lenOutput
}
