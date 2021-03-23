package perceptron

import (
	"fmt"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/param"
)

// Init
func (p *perceptron) Init(data ...interface{}) (err error) {
	if len(data) > 0 {
		switch value := data[0].(type) {
		case gonn.Filer:
			if len(p.Weights) > 0 {
				p.initFromWeight()
			}
			p.config = value
		case int:
			if len(data) == 2 {
				if v, ok := data[1].(int); ok {
					p.initFromNew(value, v)
				}
			}
		default:
			err = fmt.Errorf("init %T %w: %v", value, gonn.ErrMissingType, value)
		}
	}
	return
}

// initFromNew initialize.
func (p *perceptron) initFromNew(lenInput, lenTarget int) {
	p.lenInput = lenInput
	p.lenOutput = lenTarget
	p.lastLayerIndex = len(p.Hidden)
	if p.lastLayerIndex > 0 && p.Hidden[0] == 0 {
		p.lastLayerIndex = 0
	}

	var layer []int
	if p.lastLayerIndex > 0 {
		layer = append(p.Hidden, p.lenOutput)
	} else {
		layer = []int{p.lenOutput}
	}
	lenLayer := len(layer)

	bias := 0
	if p.Bias {
		bias = 1
	}
	biasInput := p.lenInput + bias
	var biasLayer int

	p.Weights = make(gonn.Float3Type, lenLayer)
	p.neuron = make([][]*neuron, lenLayer)
	for i, v := range layer {
		p.Weights[i] = make([][]float64, v)
		p.neuron[i] = make([]*neuron, v)
		if i > 0 {
			biasLayer = layer[i-1] + bias
		}

		for j := 0; j < v; j++ {
			if i > 0 {
				p.Weights[i][j] = make([]float64, biasLayer)
			} else {
				p.Weights[i][j] = make([]float64, biasInput)
			}
			for k := range p.Weights[i][j] {
				p.Weights[i][j][k] = param.GetRandFloat()
			}
			p.neuron[i][j] = &neuron{}
		}
	}
	p.isInit = true
}

// initFromWeight
func (p *perceptron) initFromWeight() {
	length := len(p.Weights)

	if !p.Bias && length > 1 && len(p.Weights[0])+1 == len(p.Weights[1][0]) {
		p.Bias = true
	}

	p.lastLayerIndex = length - 1
	p.lenOutput = len(p.Weights[p.lastLayerIndex])
	p.lenInput = len(p.Weights[0][0])
	if p.Bias {
		p.lenInput -= 1
	}

	if p.lastLayerIndex > 0 {
		p.Hidden = make([]int, p.lastLayerIndex)
		for i := range p.Hidden {
			p.Hidden[i] = len(p.Weights[i])
		}
	} else {
		p.Hidden = []int{0}
	}

	p.neuron = make([][]*neuron, length)
	for i, v := range p.Weights {
		p.neuron[i] = make([]*neuron, len(v))
		for j := range v {
			p.neuron[i][j] = &neuron{}
		}
	}
	p.isInit = true
}
