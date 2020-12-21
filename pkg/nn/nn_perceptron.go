package nn

import (
	"bytes"
	"fmt"
	"math"
)

const perceptronName = "perceptron"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*perceptron)(nil)

// perceptron
type perceptron struct {
	Parameter `json:"-" xml:"-"`

	// Neural network architecture name
	Name string `json:"name" xml:"name"`

	// The neuron bias, false or true
	Bias bool `json:"bias" xml:"bias"`

	// Array of the number of neurons in each hidden layer
	Hidden []int `json:"hidden" xml:"hidden>layer"`

	// Activation function mode
	Activation uint8 `json:"activation" xml:"activation"`

	// The mode of calculation of the total error
	Loss uint8 `json:"loss" xml:"loss"`

	// Minimum (sufficient) limit of the average of the error during training
	Limit float64 `json:"limit" xml:"limit"`

	// Learning coefficient, from 0 to 1
	Rate FloatType `json:"rate" xml:"rate"`

	// Weight value
	Weights Float3Type `json:"weights" xml:"weights>weights,omitempty"`

	// Neuron
	neuron [][]*neuronPerceptron

	lenInput       int
	lenOutput      int
	lastLayerIndex int

	// State of the neural network
	isInit bool // Initializing flag

	// Config
	jsonName string
}

// neuronPerceptron
type neuronPerceptron struct {
	value FloatType
	miss  FloatType
}

// Perceptron return perceptron neural network
func Perceptron() *perceptron {
	return &perceptron{
		Name:       perceptronName,
		Bias:       false,
		Hidden:     []int{},
		Activation: ModeSIGMOID,
		Loss:       ModeMSE,
		Limit:      .1,
		Rate:       FloatType(DefaultRate),
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
	return p.Hidden
}

// SetHiddenLayer
func (p *perceptron) SetHiddenLayer(layer ...int) {
	p.Hidden = layer
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
func (p *perceptron) LearningRate() float64 {
	return float64(p.Rate)
}

// SetLearningRate
func (p *perceptron) SetLearningRate(rate float64) {
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

// Train training neural network
func (p *perceptron) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if !p.isInit {
		if p.isInit = p.init(len(input), len(target[0])); !p.isInit {
			LogError(fmt.Errorf("train: %w", ErrInit))
			return -1, 0
		}
	}
	//fmt.Println("p.isInit")
	if len(target) > 0 {
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
		LogError(ErrNoTarget)
		return -1, 0
	}
	fmt.Println("Train")
	return
}

// Verify verifying neural network
func (p *perceptron) Verify(input []float64, target ...[]float64) (loss float64) {
	if !p.isInit {
		if p.isInit = p.init(len(input), len(target[0])); !p.isInit {
			LogError(fmt.Errorf("verify: %w", ErrInit))
			return -1
		}
	}
	if len(target) > 0 {
		p.calcNeuron(input)
		loss = p.calcLoss(target[0])
	} else {
		LogError(ErrNoTarget)
		return -1
	}
	return
}

// Query querying neural network
func (p *perceptron) Query(input []float64) (output []float64) {
	if !p.isInit {
		LogError(fmt.Errorf("query: %w", ErrInit))
		return nil
	}
	p.calcNeuron(input)
	output = make([]float64, p.lenOutput)
	for i, n := range p.neuron[p.lastLayerIndex] {
		output[i] = float64(n.value)
	}
	return
}

func (p *perceptron) Init() {

}

// init initialize
func (p *perceptron) init(lenInput, lenTarget int) bool {
	p.lenInput = lenInput
	p.lenOutput = lenTarget
	p.lastLayerIndex = len(p.Hidden)
	layer := append(p.Hidden, p.lenOutput)
	lenLayer := len(layer)

	bias := 0
	if p.Bias {
		bias = 1
	}
	biasInput := p.lenInput + bias
	var biasLayer int

	p.Weights = make(Float3Type, lenLayer)
	p.neuron = make([][]*neuronPerceptron, lenLayer)
	//p.input = make([]float64, lenInput)
	//p.target = make([]float64, lenTarget)

	for i, v := range layer {
		p.Weights[i] = make([][]FloatType, v)
		p.neuron[i] = make([]*neuronPerceptron, v)
		if i > 0 {
			biasLayer = layer[i-1] + bias
		}
		for j := 0; j < v; j++ {
			if i > 0 {
				p.Weights[i][j] = make([]FloatType, biasLayer)
			} else {
				p.Weights[i][j] = make([]FloatType, biasInput)
			}
			for k := range p.Weights[i][j] {
				p.Weights[i][j][k] = .5 //getRand()
			}
			p.neuron[i][j] = &neuronPerceptron{}
		}
	}
	return true
}

// initFromWeight
func (p *perceptron) initFromWeight() {

}

// initNeuronFromWeight
func (p *perceptron) initNeuronFromWeight() {
	p.neuron = make([][]*neuronPerceptron, len(p.Weights))
	for i, v := range p.Weights {
		p.neuron[i] = make([]*neuronPerceptron, len(v))
		for j := range v {
			p.neuron[i][j] = &neuronPerceptron{}
		}
	}
}

// initHiddenFromWeight
func (p *perceptron) initHiddenFromWeight() {
	length := len(p.Weights) - 1
	if len(p.Hidden) != length {
		p.Hidden = make([]int, length)
		for i := range p.Hidden {
			p.Hidden[i] = len(p.Weights[i])
		}
	}
}

// setBiasFromWeight
func (p *perceptron) setBiasFromWeight() {
	if !p.Bias {
		if len(p.Weights) > 1 {
			if len(p.Weights[0])+1 == len(p.Weights[1][0]) {
				p.Bias = true
			} else {

			}
		} else {
			LogError(fmt.Errorf("there are no hidden layers to determine the bias"))
		}
	}
}

// lenInputFromWeight
func (p *perceptron) lenInputFromWeight() {
	p.lenInput = len(p.Weights[0][0])
	if p.Bias {
		p.lenInput -= 1
	}
}

// lenOutputFromWeight
func (p *perceptron) lenOutputFromWeight() {
	p.lastLayerIndex = len(p.Weights) - 1
	p.lenOutput = len(p.Weights[p.lastLayerIndex])
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
							n.value += FloatType(input[k]) * w
						}
					} else {
						n.value += w
					}
				}
				n.value = FloatType(calcActivation(float64(n.value), p.Activation))
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
		//fmt.Printf("%4.6f\n",n.miss)
		n.miss = FloatType(target[i]) - n.value
		//fmt.Println(n.miss)
		switch p.Loss {
		default:
			fallthrough
		case ModeMSE, ModeRMSE:
			loss += math.Pow(float64(n.miss), 2)
		case ModeARCTAN:
			loss += math.Pow(math.Atan(float64(n.miss)), 2)
		}
		n.miss *= FloatType(calcDerivative(float64(n.miss), p.Activation))
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
				n.miss *= FloatType(calcDerivative(float64(n.value), p.Activation))
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
			go func(i, j, dec, length int, grad FloatType, w []FloatType) {
				for k := range w {
					if k < length {
						if i > 0 {
							p.Weights[i][j][k] += p.neuron[dec][k].value * grad
						} else {
							p.Weights[i][j][k] += FloatType(input[k]) * grad
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

// readJSON
/*func (p *perceptron) readJSON(value interface{}) {
	if b, err := json.Marshal(&value); err != nil {
		LogError(fmt.Errorf("read marshal %w", err))
	} else if err = json.Unmarshal(b, &p); err != nil {
		LogError(fmt.Errorf("read unmarshal %w", err))
	}
	p.reInit()
	if err := p.pasteWeight(); err != nil {
		LogError(fmt.Errorf("read json: %w", err))
	}
}*/

// writeReport report of neural network training results in io.Writer
func (p *perceptron) writeReport(rep *report) {
	s := "----------------------------------------------\n"
	n := "\n"
	m := "\n\n"
	b := bytes.NewBufferString("Report of Perceptron Neural Network\n\n")

	printFormat := func(format string, a ...interface{}) {
		if _, err := fmt.Fprintf(b, format, a...); err != nil {
			LogError(fmt.Errorf("write report error: %w", err))
		}
	}

	// Input layer
	if in, ok := rep.args[0].([]float64); ok {
		printFormat("%s0 Input layer size: %d\n%sNeurons:\t", s, p.lenInput, s)
		for _, v := range in {
			printFormat("  %v", v)
		}
		printFormat("%s", m)
	}

	// Layers: neuron, miss
	var t string
	for i, v := range p.neuron {
		switch i {
		case p.lastLayerIndex:
			t = "Output layer"
		default:
			t = "Hidden layer"
		}
		printFormat("%s%d %s size: %d\n%sNeurons:\t", s, i+1, t, len(p.neuron[i]), s)
		for _, w := range v {
			printFormat("  %11.8f", w.value)
		}
		printFormat("\nMiss:\t\t")
		for _, w := range v {
			printFormat("  %11.8f", w.miss)
		}
		printFormat("%s", m)
	}

	// Axons: weight
	printFormat("%sAxons (weights)\n%s", s, s)
	for _, u := range p.Weights {
		for i, v := range u {
			printFormat("%d", i+1)
			for _, w := range v {
				printFormat("\t%11.8f", w)
			}
			printFormat("%s", n)
		}
		printFormat("%s", n)
	}

	// Resume
	if loss, ok := rep.args[1].(float64); ok {
		printFormat("%sTotal loss (error):\t\t%v\n", s, loss)
	}
	if count, ok := rep.args[2].(int); ok {
		printFormat("Number of iteration:\t%v\n", count)
	}

	if _, err := b.WriteTo(rep.file); err != nil {
		LogError(fmt.Errorf("write report error: %w", err))
	} else if err = rep.file.Close(); err != nil {
		LogError(fmt.Errorf("write report close error: %w", err))
	}
}
