package perceptron

import (
	"github.com/teratron/gonn"
	param "github.com/teratron/gonn/parameter"
)

// Declare conformity with Parameter interface
/*var _ Parameter = (*perceptron)(nil)

// Parameter
type Parameter interface {
	HiddenLayer() []int
	SetHiddenLayer(...int)

	NeuronBias() bool
	SetNeuronBias(bool)

	ActivationMode() uint8
	SetActivationMode(uint8)

	LossMode() uint8
	SetLossMode(uint8)

	LossLimit() float64
	SetLossLimit(float64)

	LearningRate() float64
	SetLearningRate(float64)
}*/

func (p *perceptron) NameNN() string {
	return p.Name
}

func (p *perceptron) SetNameNN(name string) {
	p.Name = name
}

func (p *perceptron) InitNN() bool {
	return p.isInit
}

func (p *perceptron) SetInitNN(state bool) {
	p.isInit = state
}

func (p *perceptron) NameJSON() string {
	return p.jsonName
}

func (p *perceptron) SetNameJSON(name string) {
	p.jsonName = name
}

func (p *perceptron) NameYAML() string {
	return p.yamlName
}

func (p *perceptron) SetNameYAML(name string) {
	p.yamlName = name
}

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
	return param.CheckHiddenLayer(p.Hidden)
}

// SetHiddenLayer
func (p *perceptron) SetHiddenLayer(layer ...int) {
	p.Hidden = param.CheckHiddenLayer(layer)
}

// ActivationMode
func (p *perceptron) ActivationMode() uint8 {
	return p.Activation
}

// SetActivationMode
func (p *perceptron) SetActivationMode(mode uint8) {
	p.Activation = param.CheckActivationMode(mode)
}

// LossMode
func (p *perceptron) LossMode() uint8 {
	return p.Loss
}

// SetLossMode
func (p *perceptron) SetLossMode(mode uint8) {
	p.Loss = param.CheckLossMode(mode)
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
	p.Rate = param.CheckLearningRate(rate)
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
