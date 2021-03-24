package perceptron

import (
	"github.com/teratron/gonn"
	"github.com/teratron/gonn/params"
)

// NeuronBias
func (p *perceptron) NeuronBias() bool {
	return p.Bias
}

// SetNeuronBias
func (p *perceptron) SetNeuronBias(bias bool) {
	p.Bias = bias
}

// HiddenLayer
func (p *perceptron) HiddenLayer() []int {
	return params.CheckHiddenLayer(p.Hidden)
}

// SetHiddenLayer
func (p *perceptron) SetHiddenLayer(layer ...int) {
	p.Hidden = params.CheckHiddenLayer(layer)
}

// ActivationMode
func (p *perceptron) ActivationMode() uint8 {
	return p.Activation
}

// SetActivationMode
func (p *perceptron) SetActivationMode(mode uint8) {
	p.Activation = params.CheckActivationMode(mode)
}

// LossMode
func (p *perceptron) LossMode() uint8 {
	return p.Loss
}

// SetLossMode
func (p *perceptron) SetLossMode(mode uint8) {
	p.Loss = params.CheckLossMode(mode)
}

// LossLimit
func (p *perceptron) LossLimit() float64 {
	return p.Limit
}

// SetLossLimit
func (p *perceptron) SetLossLimit(limit float64) {
	p.Limit = limit
}

// LearningRate
func (p *perceptron) LearningRate() float64 {
	return p.Rate
}

// SetLearningRate
func (p *perceptron) SetLearningRate(rate float64) {
	p.Rate = params.CheckLearningRate(rate)
}

// Weight
func (p *perceptron) Weight() gonn.Floater {
	return &p.Weights
}

// SetWeight
func (p *perceptron) SetWeight(weight gonn.Floater) {
	if w, ok := weight.(gonn.Float3Type); ok {
		p.Weights = w
	}
}
