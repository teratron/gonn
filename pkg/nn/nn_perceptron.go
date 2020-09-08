// Perceptron Neural Network
package nn

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"math"

	"github.com/zigenzoog/gonn/pkg"
)

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*perceptron)(nil)

type perceptron struct {
	Architecture `json:"-" xml:"-"`
	Parameter    `json:"-" xml:"-"`
	Constructor  `json:"-" xml:"-"`

	// Configuration
	Conf struct {
		// Array of the number of neurons in each hidden layer
		HiddenLayer HiddenType `json:"hiddenLayer" xml:"hiddenLayer>layer"`

		// The neuron bias, false or true
		Bias biasType `json:"bias" xml:"bias"`

		// Activation function mode
		ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

		// The mode of calculation of the total error
		LossMode uint8 `json:"lossMode" xml:"lossMode"`

		// Minimum (sufficient) level of the average of the error during training
		LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

		// Learning coefficient, from 0 to 1
		Rate floatType `json:"rate" xml:"rate"`

		// Buffer of weight values
		Weight float3Type `json:"weight" xml:"weight>weight"`
	} `json:"perceptron,omitempty" xml:"perceptron,omitempty"`

	// Matrix
	neuron [][]*neuron
	axon   [][][]*axon
	*weight

	lastIndexLayer int
	lenInput       int
	lenOutput      int
}

// Perceptron
func Perceptron() *perceptron {
	return &perceptron{}
}

// architecture
func (p *perceptron) architecture() Architecture {
	return p.Architecture
}

// setArchitecture
func (p *perceptron) setArchitecture(network Architecture) {
	if n, ok := network.(*nn); ok {
		p.Architecture = n
	}
	p.Conf.HiddenLayer    = HiddenType{9, 2}
	p.Conf.Bias           = false
	p.Conf.ActivationMode = ModeSIGMOID
	p.Conf.LossMode       = ModeMSE
	p.Conf.LossLevel      = .0001
	p.Conf.Rate           = floatType(DefaultRate)
}

// HiddenLayer
func (p *perceptron) HiddenLayer() []uint {
	return p.Conf.HiddenLayer
}

// Bias
func (p *perceptron) Bias() bool {
	return bool(p.Conf.Bias)
}

// ActivationMode
func (p *perceptron) ActivationMode() uint8 {
	return p.Conf.ActivationMode
}

// LossMode
func (p *perceptron) LossMode() uint8 {
	return p.Conf.LossMode
}

// LossLevel
func (p *perceptron) LossLevel() float64 {
	return p.Conf.LossLevel
}

// Rate
func (p *perceptron) Rate() float32 {
	return float32(p.Conf.Rate)
}

// Weight
func (p *perceptron) Weight() Floater {
	return p.getWeight()
}

// Set
func (p *perceptron) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		case HiddenType:
			p.Conf.HiddenLayer = v
		case biasType:
			p.Conf.Bias = v
		case activationModeType:
			p.Conf.ActivationMode = uint8(v)
		case lossModeType:
			p.Conf.LossMode = uint8(v)
		case lossLevelType:
			p.Conf.LossLevel = float64(v)
		case rateType:
			p.Conf.Rate = floatType(v)
		case *weight:
			p.setWeight(v.buffer.(*float3Type))
		default:
			errNN(fmt.Errorf("%T %w for perceptron", v, ErrMissingType))
		}
	} else {
		errNN(fmt.Errorf("%w set for perceptron", ErrEmpty))
	}
}

// Get
func (p *perceptron) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		switch args[0].(type) {
		case HiddenType:
			return p.Conf.HiddenLayer
		case biasType:
			return p.Conf.Bias
		case activationModeType:
			return activationModeType(p.Conf.ActivationMode)
		case lossModeType:
			return lossModeType(p.Conf.LossMode)
		case lossLevelType:
			return lossLevelType(p.Conf.LossLevel)
		case rateType:
			return p.Conf.Rate
		case *weight:
			return p.getWeight()
		default:
			errNN(fmt.Errorf("%w for perceptron", ErrMissingType))
			return nil
		}
	} else {
		if a, ok := p.Architecture.(*nn); ok {
			return a
		}
	}
	return p
}

// Copy
func (p *perceptron) Copy(copier pkg.Copier) {
	switch c := copier.(type) {
	case *weight:
		p.copyWeight()
	default:
		errNN(fmt.Errorf("%v for copy: %v", ErrMissingType, c))
	}
}

// Paste
func (p *perceptron) Paste(paster pkg.Paster) {
	var err error
	switch v := paster.(type) {
	case *weight:
		err = p.pasteWeight()
	default:
		errNN(fmt.Errorf("%v for paste: %v", ErrMissingType, v))
	}
	if err != nil {
		// TODO: error
	}
}

// Read
func (p *perceptron) Read(reader pkg.Reader) {
	switch r := reader.(type) {
	default:
		errNN(fmt.Errorf("%v for read: %v", ErrMissingType, r))
	}
}

// Write
func (p *perceptron) Write(writer ...pkg.Writer) {
	for _, w := range writer {
		switch v := w.(type) {
		case *report:
			p.writeReport(v)
		default:
			errNN(fmt.Errorf("%v for write: %v", ErrMissingType, w))
		}
	}
}

// init initialize
func (p *perceptron) init(lenInput int, lenTarget ...interface{}) bool {
	if len(lenTarget) > 0 {
		var tmp HiddenType
		defer func() {
			tmp = nil
		}()

		p.lastIndexLayer = len(p.Conf.HiddenLayer)
		p.lenInput = lenInput
		p.lenOutput = lenTarget[0].(int)
		tmp = append(p.Conf.HiddenLayer, uint(p.lenOutput))
		layer := make(HiddenType, p.lastIndexLayer+1)
		lenLayer := copy(layer, tmp)

		bias := 0
		if p.Conf.Bias {
			bias = 1
		}

		p.neuron = make([][]*neuron, lenLayer)
		p.axon = make([][][]*axon, lenLayer)
		for i, l := range layer {
			p.neuron[i] = make([]*neuron, l)
			p.axon[i]   = make([][]*axon, l)
			for j := 0; j < int(l); j++ {
				if i == 0 {
					p.axon[i][j] = make([]*axon, p.lenInput + bias)
				} else {
					p.axon[i][j] = make([]*axon, int(layer[i - 1]) + bias)
				}
			}
		}
		p.initNeuron()
		p.initAxon()

		return true
	} else {
		errNN(ErrNoTarget)
		return false
	}
}

// reInit
func (p *perceptron) reInit() {
	bias := 0
	if p.Conf.Bias {
		bias = 1
	}
	length := len(p.Conf.Weight) - 1
	p.Conf.HiddenLayer = make(HiddenType, length)
	for i := range p.Conf.HiddenLayer {
		p.Conf.HiddenLayer[i] = uint(len(p.Conf.Weight[i]))
	}
	if n, ok := p.Architecture.(*nn); ok {
		n.IsInit = p.init(len(p.Conf.Weight[0][0]) - bias, len(p.Conf.Weight[length]))
	}
}

// initNeuron
func (p *perceptron) initNeuron() {
	for i, v := range p.neuron {
		for j := range v {
			p.neuron[i][j] = &neuron{
				axon:     p.axon[i][j],
				specific: floatType(0),
			}
		}
	}
}

// initAxon
func (p *perceptron) initAxon() {
	isTrain := true
	if n, ok := p.Architecture.(*nn); ok && !n.IsTrain {
		isTrain = false
	}
	for i, v := range p.axon {
		for j, w := range v {
			for k := range w {
				p.axon[i][j][k] = &axon{
					synapse: map[string]Synapser{},
				}
				if !isTrain {
					p.axon[i][j][k].weight = .5 //getRand()
				}
				if i == 0 {
					if k < p.lenInput {
						p.axon[i][j][k].synapse["input"] = floatType(0)
					} else {
						p.axon[i][j][k].synapse["input"] = biasType(true)
					}
				} else {
					if k < len(p.axon[i-1]) {
						p.axon[i][j][k].synapse["input"] = p.neuron[i - 1][k]
					} else {
						p.axon[i][j][k].synapse["input"] = biasType(true)
					}
				}
				p.axon[i][j][k].synapse["output"] = p.neuron[i][j]
			}
		}
	}
}

// initSynapseInput
func (p *perceptron) initSynapseInput(input []float64) {
	for j, w := range p.axon[0] {
		for k := range w {
			if k < p.lenInput {
				p.axon[0][j][k].synapse["input"] = floatType(input[k])
			}
		}
	}
}

// calcNeuron
func (p *perceptron) calcNeuron(input []float64) {
	p.initSynapseInput(input)
	wait := make(chan bool)
	defer close(wait)
	for _, v := range p.neuron {
		for _, w := range v {
			go func(n *neuron) {
				n.value = 0
				for _, a := range n.axon {
					n.value += a.getSynapseInput() * a.weight
				}
				n.value = floatType(calcActivation(float64(n.value), p.Conf.ActivationMode))
				wait <- true
			}(w)
		}
		for range v {
			<-wait
		}
	}
}

// calcLoss calculating the error of the output neuron
func (p *perceptron) calcLoss(target []float64) (loss float64) {
	for i, v := range p.neuron[p.lastIndexLayer] {
		if miss, ok := v.specific.(floatType); ok {
			miss = floatType(target[i]) - v.value
			switch p.Conf.LossMode {
			default:
				fallthrough
			case ModeMSE, ModeRMSE:
				loss += math.Pow(float64(miss), 2)
			case ModeARCTAN:
				loss += math.Pow(math.Atan(float64(miss)), 2)
			}
			miss *= floatType(calcDerivative(float64(v.value), p.Conf.ActivationMode))
			v.specific = miss
		}
	}
	loss /= float64(p.lenOutput)
	if p.Conf.LossMode == ModeRMSE {
		loss = math.Sqrt(loss)
	}
	return
}

// calcMiss calculating the error of neurons in hidden layers
func (p *perceptron) calcMiss(input []float64) {
	wait := make(chan bool)
	defer close(wait)
	for i := p.lastIndexLayer - 1; i >= 0; i-- {
		for j, v := range p.neuron[i] {
			go func(j int, n *neuron) {
				if miss, ok := n.specific.(floatType); ok {
					miss = 0
					for _, w := range p.neuron[i + 1] {
						if m, ok := w.specific.(floatType); ok {
							miss += m * w.axon[j].weight
						}
					}
					miss *= floatType(calcDerivative(float64(n.value), p.Conf.ActivationMode))
					n.specific = miss
				}
				wait <- true
			}(j, v)
		}
		for range p.neuron[i] {
			<-wait
		}
	}
}

// calcAxon update weights
func (p *perceptron) calcAxon(input []float64) {
	p.calcMiss(input)
	wait := make(chan bool)
	defer close(wait)
	for _, u := range p.axon {
		for _, v := range u {
			for _, w := range v {
				go func(a *axon) {
					if n, ok := a.synapse["output"].(*neuron); ok {
						if miss, ok := n.specific.(floatType); ok {
							a.weight += a.getSynapseInput() * miss * p.Conf.Rate
						}
					}
					wait <- true
				}(w)
			}
			for range v {
				<-wait
			}
		}
	}
}

// Train training neural network
func (p *perceptron) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if len(target) > 0 {
		for count < 1 /*MaxIteration*/ {
			p.calcNeuron(input)
			if loss = p.calcLoss(target[0]); loss <= p.Conf.LossLevel || loss <= MinLossLevel {
				break
			}
			p.calcMiss(input)
			p.calcAxon(input)
			count++
		}
	} else {
		errNN(ErrNoTarget)
		return -1, 0
	}
	return
}

// Query querying neural network
func (p *perceptron) Query(input []float64) (output []float64) {
	p.calcNeuron(input)
	output = make([]float64, p.lenOutput)
	for i, n := range p.neuron[p.lastIndexLayer] {
		output[i] = float64(n.value)
	}
	return
}

// Verify verifying neural network
func (p *perceptron) Verify(input []float64, target ...[]float64) (loss float64) {
	if len(target) > 0 {
		p.calcNeuron(input)
		loss = p.calcLoss(target[0])
	} else {
		errNN(ErrNoTarget)
		return -1
	}
	return
}

// initWeight
func (p *perceptron) initWeight() {
	p.Conf.Weight = make(float3Type, len(p.axon))
	for i, v := range p.axon {
		p.Conf.Weight[i] = make(float2Type, len(p.axon[i]))
		for j := range v {
			p.Conf.Weight[i][j] = make(float1Type, len(p.axon[i][j]))
		}
	}
	p.weight = &weight{
		isInitWeight: true,
		buffer:       &p.Conf.Weight,
	}
}

// getWeight
func (p *perceptron) getWeight() *float3Type {
	p.copyWeight()
	return &p.Conf.Weight
}

// setWeight
func (p *perceptron) setWeight(weight *float3Type) {
	for i, u := range *weight {
		for j, v := range u {
			for k, w := range v {
				p.axon[i][j][k].weight = w
			}
		}
	}
}

// copyWeight copies weights to the buffer
func (p *perceptron) copyWeight() {
	if p.weight == nil {
		p.initWeight()
	}
	for i, u := range p.axon {
		for j, v := range u {
			for k, w := range v {
				p.Conf.Weight[i][j][k] = w.weight
			}
		}
	}
}

// pasteWeight inserts weights from the buffer
func (p *perceptron) pasteWeight() (err error) {
	if p.Conf.Weight != nil {
		for i, u := range p.Conf.Weight {
			for j, v := range u {
				for k, w := range v {
					p.axon[i][j][k].weight = w
				}
			}
		}
		p.deleteWeight()
	}
	// TODO: where error?
	return
}

// deleteWeight
func (p *perceptron) deleteWeight() {
	p.Conf.Weight = nil
	p.weight      = nil
}

// readJSON
func (p *perceptron) readJSON(value interface{}) {
	if b, err := json.Marshal(&value); err != nil {
		errJSON(fmt.Errorf("read marshal %w", err))
	} else if err = json.Unmarshal(b, &p.Conf); err != nil {
		errJSON(fmt.Errorf("read unmarshal %w", err))
	}
	p.reInit()
	if err := p.pasteWeight(); err != nil {
		errNN(err)
	}
}

// writeReport report of neural network training results in io.Writer
func (p *perceptron) writeReport(report *report) {
	s := "----------------------------------------------\n"
	n := "\n"
	m := "\n\n"
	b := bytes.NewBufferString("Report of Perceptron Neural Network\n\n")

	printFormat := func(format string, a ...interface{}) {
		if _, err := fmt.Fprintf(b, format, a...); err != nil {
			errNN(err)
		}
	}

	// Input layer
	if in, ok := report.args[0].([]float64); ok {
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
		case p.lastIndexLayer:
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
			printFormat("  %11.8f", w.specific)
		}
		printFormat("%s", m)
	}

	// Axons: weight
	printFormat("%sAxons (weights)\n%s", s, s)
	for _, u := range p.axon {
		for i, v := range u {
			printFormat("%d", i+1)
			for _, w := range v {
				printFormat("\t%11.8f", w.weight)
			}
			printFormat("%s", n)
		}
		printFormat("%s", n)
	}

	// Resume
	if loss, ok := report.args[1].(float64); ok {
		printFormat("%sTotal loss (error):\t\t%v\n", s, loss)
	}
	if count, ok := report.args[2].(int); ok {
		printFormat("Number of iteration:\t%v\n", count)
	}

	if _, err := b.WriteTo(report.file); err != nil {
		log.Println(err)
	} else if err = report.file.Close(); err != nil {
		log.Println(err)
	}
}
