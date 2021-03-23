package perceptron

import (
	"github.com/teratron/gonn"
	"github.com/teratron/gonn/param"
)

/*func (p *perceptron) NameNN() string {
	return p.Name
}

func (p *perceptron) SetNameNN(name string) {
	p.Name = name
}*/

/*func (p *perceptron) InitNN() bool {
	return p.isInit
}

func (p *perceptron) SetInitNN(state bool) {
	p.isInit = state
}*/

// SetConfig
/*func (p *perceptron) SetConfig(file gonn.Filer) {
	p.config = file
	switch cfg := file.(type) {
	case *util.FileJSON:
		p.jsonConfig = cfg
	case *util.FileYAML:
		p.yamlConfig = cfg
	default:
		log.Println(fmt.Errorf("set config: %T %w: %v", cfg, gonn.ErrMissingType, cfg))
	}
}*/

/*func (p *perceptron) NameJSON() string {
	return p.jsonConfig.Name
}

func (p *perceptron) SetNameJSON(name string) {
	p.jsonConfig.Name = name
}

func (p *perceptron) NameYAML() string {
	return p.yamlConfig.Name
}

func (p *perceptron) SetNameYAML(name string) {
	p.yamlConfig.Name = name
}*/

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
