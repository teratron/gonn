package nn

import (
	"fmt"
	"math"
)

const perceptronName = "perceptron"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*perceptron)(nil)

// perceptron
type perceptron struct {
	Parameter `json:"-"`

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
	Rate floatType `json:"rate"`

	// Weight value
	Weights Float3Type `json:"weights,omitempty"`

	// Neuron
	neuron [][]*neuronPerceptron

	// Settings
	lenInput       int
	lenOutput      int
	lastLayerIndex int
	isInit         bool
	jsonName       string
}

// neuronPerceptron
type neuronPerceptron struct {
	value floatType
	miss  floatType
}

// Perceptron return perceptron neural network
func Perceptron() *perceptron {
	return &perceptron{
		Name:       perceptronName,
		Activation: ModeSIGMOID,
		Loss:       ModeMSE,
		Limit:      .1,
		Rate:       floatType(DefaultRate),
	}
}

func (p *perceptron) name() string {
	return p.Name
}

func (p *perceptron) setName(name string) {
	p.Name = name
}

func (p *perceptron) stateInit() bool {
	return p.isInit
}

func (p *perceptron) setStateInit(state bool) {
	p.isInit = state
}

func (p *perceptron) nameJSON() string {
	return p.jsonName
}

func (p *perceptron) setNameJSON(name string) {
	p.jsonName = name
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
	return checkHiddenLayer(p.Hidden)
}

// SetHiddenLayer
func (p *perceptron) SetHiddenLayer(layer ...int) {
	p.Hidden = checkHiddenLayer(layer)
}

// ActivationMode
func (p *perceptron) ActivationMode() uint8 {
	return p.Activation
}

// SetActivationMode
func (p *perceptron) SetActivationMode(mode uint8) {
	p.Activation = checkActivationMode(mode)
}

// LossMode
func (p *perceptron) LossMode() uint8 {
	return p.Loss
}

// SetLossMode
func (p *perceptron) SetLossMode(mode uint8) {
	p.Loss = checkLossMode(mode)
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
func (p *perceptron) LearningRate() float32 {
	return float32(p.Rate)
}

// SetLearningRate
func (p *perceptron) SetLearningRate(rate float32) {
	p.Rate = checkLearningRate(rate)
}

// Weight
func (p *perceptron) Weight() Floater {
	return &p.Weights
}

// SetWeight
func (p *perceptron) SetWeight(weight Floater) {
	if w, ok := weight.(Float3Type); ok {
		p.Weights = w
	}
}

// Read
func (p *perceptron) Read(reader Reader) {
	switch r := reader.(type) {
	case Filer:
		r.Read(p)
		if len(p.Weights) > 0 {
			p.initFromWeight()
		}
		switch s := r.(type) {
		case jsonString:
			p.jsonName = string(s)
		default:
			LogError(fmt.Errorf("%T %w for file: %v", s, ErrMissingType, s))
		}
	default:
		LogError(fmt.Errorf("%T %w for read: %v", r, ErrMissingType, r))
	}
}

// Write
func (p *perceptron) Write(writer ...Writer) {
	if len(writer) > 0 {
		for _, w := range writer {
			switch v := w.(type) {
			case Filer:
				v.Write(p)
			case *report:
				p.writeReport(v)
			default:
				LogError(fmt.Errorf("%T %w for write: %v", v, ErrMissingType, w))
			}
		}
	} else {
		LogError(fmt.Errorf("%w for write", ErrEmpty))
	}
}

// Train training dataset
func (p *perceptron) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if len(input) > 0 {
		if len(target) > 0 && len(target[0]) > 0 {
			if !p.isInit {
				p.initFromNew(len(input), len(target[0]), getRandFloat)
				if !p.isInit {
					LogError(fmt.Errorf("train: %w", ErrInit))
					return -1, 0
				}
			} else {
				if p.lenInput != len(input) {
					LogError(fmt.Errorf("train: invalid number of elements in the input data"))
					return -1, 0
				}
				if p.lenOutput != len(target[0]) {
					LogError(fmt.Errorf("train: invalid number of elements in the target data"))
					return -1, 0
				}
			}
			for count < 1 /*MaxIteration*/ {
				//fmt.Println(count, loss, "0")
				p.calcNeuron(input)
				//fmt.Println(count, loss, "1", p.Limit)
				if loss = p.calcLoss(target[0]); loss <= p.Limit {
					break
				}
				//fmt.Println(count, loss, "2")
				p.calcMiss()
				//fmt.Println(count, loss, "3")
				p.updWeight(input)
				//fmt.Println(count, loss, "4")
				count++
			}
		} else {
			LogError(fmt.Errorf("train: %w", ErrNoTarget))
			return -1, 0
		}
	} else {
		LogError(fmt.Errorf("train: %w", ErrNoInput))
		return -1, 0
	}
	//fmt.Println("Train")
	return
}

// Verify verifying dataset
func (p *perceptron) Verify(input []float64, target ...[]float64) (loss float64) {
	if len(input) > 0 {
		if len(target) > 0 && len(target[0]) > 0 {
			if !p.isInit {
				p.initFromNew(len(input), len(target[0]), getRandFloat)
				if !p.isInit {
					LogError(fmt.Errorf("verify: %w", ErrInit))
					return -1
				}
			} else {
				if p.lenInput != len(input) {
					LogError(fmt.Errorf("verify: invalid number of elements in the input data"))
					return -1
				}
				if p.lenOutput != len(target[0]) {
					LogError(fmt.Errorf("verify: invalid number of elements in the target data"))
					return -1
				}
			}
			p.calcNeuron(input)
			loss = p.calcLoss(target[0])
		} else {
			LogError(fmt.Errorf("verify: %w", ErrNoTarget))
			return -1
		}
	} else {
		LogError(fmt.Errorf("verify: %w", ErrNoInput))
		return -1
	}
	return
}

// Query querying dataset
func (p *perceptron) Query(input []float64) (output []float64) {
	if len(input) > 0 {
		if !p.isInit {
			LogError(fmt.Errorf("query: %w", ErrInit))
			return nil
		} else if p.lenInput != len(input) {
			LogError(fmt.Errorf("query: invalid number of elements in the input data"))
			return nil
		}
		p.calcNeuron(input)
		output = make([]float64, p.lenOutput)
		for i, n := range p.neuron[p.lastLayerIndex] {
			output[i] = float64(n.value)
		}
	} else {
		LogError(fmt.Errorf("query: %w", ErrNoInput))
		return nil
	}
	return
}

// initFromNew initialize
func (p *perceptron) initFromNew(lenInput, lenTarget int, random func() floatType) {
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

	p.Weights = make(Float3Type, lenLayer)
	p.neuron = make([][]*neuronPerceptron, lenLayer)
	for i, v := range layer {
		p.Weights[i] = make([][]floatType, v)
		p.neuron[i] = make([]*neuronPerceptron, v)
		if i > 0 {
			biasLayer = layer[i-1] + bias
		}
		for j := 0; j < v; j++ {
			if i > 0 {
				p.Weights[i][j] = make([]floatType, biasLayer)
			} else {
				p.Weights[i][j] = make([]floatType, biasInput)
			}
			for k := range p.Weights[i][j] {
				p.Weights[i][j][k] = random() //.5//p.random() //randFloat() //.5 //getRandFloat()
			}
			p.neuron[i][j] = &neuronPerceptron{}
		}
	}
	p.isInit = true
}

// initFromWeight
func (p *perceptron) initFromWeight() {
	length := len(p.Weights)

	if !p.Bias {
		if length > 1 {
			if len(p.Weights[0])+1 == len(p.Weights[1][0]) {
				p.Bias = true
			}
		}
	}

	p.lastLayerIndex = length - 1
	p.lenOutput = len(p.Weights[p.lastLayerIndex])
	p.lenInput = len(p.Weights[0][0])
	if p.Bias {
		p.lenInput -= 1
	}

	p.Hidden = make([]int, p.lastLayerIndex)
	for i := range p.Hidden {
		p.Hidden[i] = len(p.Weights[i])
	}

	p.neuron = make([][]*neuronPerceptron, length)
	for i, v := range p.Weights {
		p.neuron[i] = make([]*neuronPerceptron, len(v))
		for j := range v {
			p.neuron[i][j] = &neuronPerceptron{}
		}
	}
	p.isInit = true
}

// calcNeuron
func (p *perceptron) calcNeuron(input []float64) {
	wait := make(chan bool)
	defer close(wait)

	var length int
	for i, v := range p.neuron {
		dec := i - 1
		if i > 0 {
			length = len(p.neuron[dec])
		} else {
			length = p.lenInput
		}
		for j, n := range v {
			go func(j int, n *neuronPerceptron) {
				n.value = 0
				for k, w := range p.Weights[i][j] {
					if k < length {
						if i > 0 {
							n.value += p.neuron[dec][k].value * w
						} else {
							n.value += floatType(input[k]) * w
						}
					} else {
						n.value += w
					}
				}
				n.value = floatType(Activation(float64(n.value), p.Activation))
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
		n.miss = floatType(target[i]) - n.value
		switch p.Loss {
		default:
			fallthrough
		case ModeMSE, ModeRMSE:
			loss += math.Pow(float64(n.miss), 2)
		case ModeARCTAN:
			loss += math.Pow(math.Atan(float64(n.miss)), 2)
		}
		n.miss *= floatType(Derivative(float64(n.miss), p.Activation))
	}
	loss /= float64(p.lenOutput)
	if p.Loss == ModeRMSE {
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
			go func(j int, n *neuronPerceptron) {
				n.miss = 0
				for k, m := range p.neuron[inc] {
					n.miss += m.miss * p.Weights[inc][k][j]
				}
				n.miss *= floatType(Derivative(float64(n.value), p.Activation))
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

	var length int
	for i, v := range p.Weights {
		dec := i - 1
		if i > 0 {
			length = len(p.neuron[dec])
		} else {
			length = p.lenInput
		}
		for j, w := range v {
			go func(i, j, dec, length int, grad floatType, w []floatType) {
				for k := range w {
					if k < length {
						if i > 0 {
							p.Weights[i][j][k] += p.neuron[dec][k].value * grad
						} else {
							p.Weights[i][j][k] += floatType(input[k]) * grad
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
