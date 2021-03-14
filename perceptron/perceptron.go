package perceptron

import (
	"fmt"
	"log"
	"math"

	"github.com/teratron/gonn"
	param "github.com/teratron/gonn/parameter"
	"github.com/teratron/gonn/util"
)

//const perceptronName = "perceptron"

// Declare conformity with NeuralNetwork interface
var _ gonn.NeuralNetwork = (*perceptron)(nil)

// perceptron
type perceptron struct {
	//Parameter `json:"-"`

	// Neural network architecture name
	Name string `json:"name"`

	// The neuron bias, false or true
	Bias bool `json:"bias"`

	// Array of the number of neurons in each hidden layer
	Hidden []int `json:"hidden,omitempty"`

	// Activation function mode
	Activation uint8 `json:"activation"`

	// The mode of calculation of the total error
	Loss uint8 `json:"loss"`

	// Minimum (sufficient) limit of the average of the error during training
	Limit float64 `json:"limit"`

	// Learning coefficient, from 0 to 1
	Rate float64 `json:"rate"`

	// Weight value
	Weights gonn.Float3Type `json:"weights,omitempty"`

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

// neuronPerceptron
type neuron struct {
	value float64
	miss  float64
}

// Perceptron return perceptron neural network
/*func Perceptron() *perceptron {
	return &perceptron{
		Name:       perceptronName,
		Activation: ModeSIGMOID,
		Loss:       ModeMSE,
		Limit:      .1,
		Rate:       DefaultRate,
	}
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

// Read
func (p *perceptron) Read(reader gonn.Reader) (err error) {
	switch r := reader.(type) {
	case gonn.Filer:
		err = r.Read(p)
		if len(p.Weights) > 0 {
			p.initFromWeight()
		}
		switch s := r.(type) {
		case util.JsonString:
			p.jsonName = string(s)
		}
	default:
		err = fmt.Errorf("%T %w: %v", r, gonn.ErrMissingType, r)
	}
	if err != nil {
		err = fmt.Errorf("perceptron read: %w", err)
	}
	return
}

// Write
func (p *perceptron) Write(writer ...gonn.Writer) (err error) {
	if len(writer) > 0 {
		for _, w := range writer {
			switch v := w.(type) {
			case gonn.Filer:
				err = v.Write(p)
			default:
				err = fmt.Errorf("%T %w: %v", v, gonn.ErrMissingType, w)
			}
		}
	} else {
		err = fmt.Errorf("%w args", gonn.ErrEmpty)
	}
	if err != nil {
		err = fmt.Errorf("perceptron write: %w", err)
	}
	return
}

// Verify verifying dataset
func (p *perceptron) Verify(input []float64, target ...[]float64) (loss float64) {
	var err error
	if len(input) > 0 {
		if len(target) > 0 && len(target[0]) > 0 {
			if !p.isInit {
				p.initFromNew(len(input), len(target[0]))
			} else {
				if p.lenInput != len(input) {
					err = fmt.Errorf("invalid number of elements in the input data")
					goto ERROR
				}
				if p.lenOutput != len(target[0]) {
					err = fmt.Errorf("invalid number of elements in the target data")
					goto ERROR
				}
			}
			p.calcNeuron(input)
			loss = p.calcLoss(target[0])
		} else {
			err = gonn.ErrNoTarget
		}
	} else {
		err = gonn.ErrNoInput
	}
ERROR:
	if err != nil {
		log.Println(fmt.Errorf("verify: %w", err))
		return -1
	}
	return
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

// calcNeuron
func (p *perceptron) calcNeuron(input []float64) {
	wait := make(chan bool)
	defer close(wait)

	var length, dec int
	for i, v := range p.neuron {
		if i > 0 {
			dec = i - 1
			length = len(p.neuron[dec])
		} else {
			length = p.lenInput
		}
		for j, n := range v {
			go func(j int, n *neuron) {
				n.value = 0
				for k, w := range p.Weights[i][j] {
					if k < length {
						if i > 0 {
							n.value += p.neuron[dec][k].value * w
						} else {
							n.value += input[k] * w
						}
					} else {
						n.value += w
					}
				}
				n.value = param.Activation(n.value, p.Activation)
				wait <- true
			}(j, n)
		}
		for range v {
			<-wait
		}
	}
}

// calcLoss calculating the error of the output neuron
func (p *perceptron) calcLoss(target []float64) (loss float64) {
	for i, n := range p.neuron[p.lastLayerIndex] {
		n.miss = target[i] - n.value
		switch p.Loss {
		default:
			fallthrough
		case param.ModeMSE, param.ModeRMSE:
			loss += math.Pow(n.miss, 2)
		case param.ModeARCTAN:
			loss += math.Pow(math.Atan(n.miss), 2)
		}
		n.miss *= param.Derivative(n.value, p.Activation)
	}
	loss /= float64(p.lenOutput)
	if p.Loss == param.ModeRMSE {
		loss = math.Sqrt(loss)
	}
	return
}

// calcMiss calculating the error of neurons in hidden layers
func (p *perceptron) calcMiss() {
	wait := make(chan bool)
	defer close(wait)

	for i := p.lastLayerIndex - 1; i >= 0; i-- {
		inc := i + 1
		for j, n := range p.neuron[i] {
			go func(j int, n *neuron) {
				n.miss = 0
				for k, m := range p.neuron[inc] {
					n.miss += m.miss * p.Weights[inc][k][j]
				}
				n.miss *= param.Derivative(n.value, p.Activation)
				wait <- true
			}(j, n)
		}
		for range p.neuron[i] {
			<-wait
		}
	}
}

// updWeight update weights
func (p *perceptron) updWeight(input []float64) {
	wait := make(chan bool)
	defer close(wait)

	var length, dec int
	for i, v := range p.Weights {
		if i > 0 {
			dec = i - 1
			length = len(p.neuron[dec])
		} else {
			length = p.lenInput
		}
		for j, w := range v {
			go func(i, j, dec, length int, grad float64, w []float64) {
				for k := range w {
					if k < length {
						if i > 0 {
							p.Weights[i][j][k] += p.neuron[dec][k].value * grad
						} else {
							p.Weights[i][j][k] += input[k] * grad
						}
					} else {
						p.Weights[i][j][k] += grad
					}
				}
				wait <- true
			}(i, j, dec, length, p.neuron[i][j].miss*p.Rate, w)
		}
	}
	for _, v := range p.Weights {
		for range v {
			<-wait
		}
	}
}
