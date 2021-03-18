package perceptron

import (
	"github.com/teratron/gonn"
	param "github.com/teratron/gonn/nn/parameter"
)

// Name of the neural network architecture.
const Name = "perceptron"

// Declare conformity with NeuralNetwork interface
var _ gonn.NeuralNetwork = (*perceptron)(nil)

type perceptron struct {
	gonn.Parameter `json:"-" yaml:"-"`

	// Neural network architecture name
	Name string `json:"name" yaml:"name"`

	// The neuron bias, false or true
	Bias bool `json:"bias" yaml:"bias"`

	// Array of the number of neurons in each hidden layer
	Hidden []int `json:"hidden,omitempty" yaml:"hidden,omitempty"`

	// Activation function mode
	Activation uint8 `json:"activation" yaml:"activation"`

	// The mode of calculation of the total error
	Loss uint8 `json:"loss" yaml:"loss"`

	// Minimum (sufficient) limit of the average of the error during training
	Limit float64 `json:"limit" yaml:"limit"`

	// Learning coefficient, from 0 to 1
	Rate float64 `json:"rate" yaml:"rate"`

	// Weight value
	Weights gonn.Float3Type `json:"weights,omitempty" yaml:"weights,omitempty"`

	// Neuron
	neuron [][]*neuron

	// Settings
	lenInput       int
	lenOutput      int
	lastLayerIndex int
	isInit         bool
	jsonName       string
	yamlName       string
}

type neuron struct {
	value float64
	miss  float64
}

// Perceptron return perceptron neural network
func Perceptron() *perceptron {
	return &perceptron{
		Name:       Name,
		Activation: param.ModeSIGMOID,
		Loss:       param.ModeMSE,
		Limit:      .1,
		Rate:       param.DefaultRate,
	}
}

// initFromNew initialize
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
