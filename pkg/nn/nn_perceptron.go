// Perceptron Neural Network
package nn

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/zigenzoog/gonn/pkg"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
)

var _ NeuralNetwork = (*perceptron)(nil)

type Perceptron interface {
	HiddenLayer() []uint
	Bias() bool
	ActivationMode() uint8
	LossMode() uint8
	LossLevel() float64
	Rate() float32
}

type perceptron struct {
	Architecture

	hiddenLayer    HiddenType // Array of the number of neurons in each hidden layer
	bias           biasType   // The neuron bias, false or true
	activationMode uint8      // Activation function mode
	lossMode       uint8      //
	lossLevel      float64    // Minimum (sufficient) level of the average of the error during training
	rate           floatType  // Learning coefficient, from 0 to 1

	neuron			[][]*Neuron
	axon			[][][]*Axon

	lastIndexLayer	int
	lenInput		int
	lenOutput		int
}

// Returns a new Perceptron neural network instance with the default parameters
func (n *NN) Perceptron() NeuralNetwork {
	n.Architecture = &perceptron{
		Architecture:   n,
		hiddenLayer:    HiddenType{},
		bias:           false,
		activationMode: ModeSIGMOID,
		lossMode:       ModeMSE,
		lossLevel:      .0001,
		rate:           floatType(DefaultRate),
	}
	return n
}

// Preset
func (p *perceptron) Preset(name string) {
	switch name {
	default:
		fallthrough
	case "default":
		p.Set(
			HiddenLayer(),
			Bias(false),
			ActivationMode(ModeSIGMOID),
			LossMode(ModeMSE),
			LossLevel(.0001),
			Rate(DefaultRate))
	}
}

// HiddenLayer
func (p *perceptron) HiddenLayer() []uint {
	return p.hiddenLayer
}

// Bias
func (p *perceptron) Bias() bool {
	return bool(p.bias)
}

// ActivationMode
func (p *perceptron) ActivationMode() uint8 {
	//fmt.Println("*")
	return p.activationMode
}

// LossMode
func (p *perceptron) LossMode() uint8 {
	return p.lossMode
}

// LossLevel
func (p *perceptron) LossLevel() float64 {
	return p.lossLevel
}

// Rate
func (p *perceptron) Rate() float32 {
	return float32(p.rate)
}

// Setter
func (p *perceptron) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		case HiddenType:
			p.hiddenLayer = v
		case biasType:
			p.bias = v
		case activationModeType:
			p.activationMode = uint8(v)
		case lossModeType:
			p.lossMode = uint8(v)
		case lossLevelType:
			p.lossLevel = float64(v)
		case rateType:
			p.rate = floatType(v)
		default:
			pkg.Log("This type is missing for Perceptron Neural Network", true) // !!!
			log.Printf("\tset: %T %v\n", v, v) // !!!
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Getter
func (p *perceptron) Get(args ...pkg.Getter) pkg.GetterSetter {
	if len(args) > 0 {
		switch args[0].(type) {
		case HiddenType:
			return p.hiddenLayer
		case biasType:
			return p.bias
		case activationModeType:
			return activationModeType(p.activationMode)
		case lossModeType:
			return lossModeType(p.lossMode)
		case lossLevelType:
			return lossLevelType(p.lossLevel)
		case rateType:
			return p.rate
		default:
			pkg.Log("This type is missing for Perceptron Neural Network", true) // !!!
			log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
			return nil
		}
	} else {
		if a, ok := p.Architecture.(NeuralNetwork); ok {
			return a
		}
	}
	return p
}

// init Initialization
func (p *perceptron) init(lenInput int, lenTarget ...interface{}) bool {
	if len(lenTarget) > 0 {
		var tmp HiddenType
		defer func() {
			tmp = nil
		}()

		p.lastIndexLayer = len(p.hiddenLayer)
		p.lenInput       = lenInput
		p.lenOutput      = lenTarget[0].(int)
		tmp              = append(p.hiddenLayer, uint(p.lenOutput))
		layer           := make(HiddenType, p.lastIndexLayer + 1)
		lenLayer        := copy(layer, tmp)

		b := 0
		if p.bias {
			b = 1
		}
		p.neuron = make([][]*Neuron, lenLayer)
		p.axon   = make([][][]*Axon, lenLayer)
		for i, l := range layer {
			p.neuron[i] = make([]*Neuron, l)
			p.axon[i]   = make([][]*Axon, l)
			for j := 0; j < int(l); j++ {
				if i == 0 {
					p.axon[i][j] = make([]*Axon, p.lenInput + b)
				} else {
					p.axon[i][j] = make([]*Axon, int(layer[i - 1]) + b)
				}
			}
		}
		if n, ok := p.Get().(*NN); ok && !n.isTrain {
			p.initNeuron()
			p.initAxon()
		}
		return true
	} else {
		pkg.Log("No target data", true) // !!!
		return false
	}
}

//
func (p *perceptron) initNeuron() {
	for i, v := range p.neuron {
		for j := range v {
			p.neuron[i][j] = &Neuron{
				axon:     p.axon[i][j],
				specific: floatType(0),
			}
		}
	}
}

//
func (p *perceptron) initAxon() {
	for i, v := range p.axon {
		for j, w := range v {
			for k := range w {
				p.axon[i][j][k] = &Axon{
					weight:  .5,//getRand(),
					synapse: map[string]pkg.Getter{},
				}
				if i == 0 {
					if k < p.lenInput {
						p.axon[i][j][k].synapse["input"] = floatType(0)
					} else {
						p.axon[i][j][k].synapse["input"] = biasType(true)
					}
				} else {
					if k < len(p.axon[i - 1]) {
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

//
func (p *perceptron) initSynapseInput(input []float64) {
	for j, w := range p.axon[0] {
		for k := range w {
			if k < p.lenInput {
				p.axon[0][j][k].synapse["input"] = floatType(input[k])
			}
		}
	}
}

func (p *perceptron) calcNeuron(input []float64) {
	p.initSynapseInput(input)
	wait := make(chan bool)
	defer close(wait)
	for _, v := range p.neuron {
		for _, w := range v {
			go func(n *Neuron) {
				n.value = 0
				for _, a := range n.axon {
					n.value += getSynapseInput(a) * a.weight
				}
				n.value = floatType(calcActivation(float64(n.value), p.activationMode))
				wait <- true
			}(w)
		}
		for range v {
			<- wait
		}
	}
}

// Calculating the error of the output neuron
func (p *perceptron) calcLoss(target []float64) (loss float64) {
	for i, v := range p.neuron[p.lastIndexLayer] {
		if miss, ok := v.specific.(floatType); ok {
			miss = floatType(target[i]) - v.value
			switch p.lossMode {
			default: fallthrough
			case ModeMSE, ModeRMSE:
				loss += math.Pow(float64(miss), 2)
			case ModeARCTAN:
				loss += math.Pow(math.Atan(float64(miss)), 2)
			}
			miss *= floatType(calcDerivative(float64(v.value), p.activationMode))
			v.specific = miss
		}
	}
	loss /= float64(p.lenOutput)
	if p.lossMode == ModeRMSE {
		loss = math.Sqrt(loss)
	}
	return
}

// Calculating the error of neurons in hidden layers and update weights
func (p *perceptron) calcMiss(input []float64) {
	wait := make(chan bool)
	defer close(wait)
	for i := p.lastIndexLayer - 1; i >= 0; i-- {
		for j, v := range p.neuron[i] {
			go func(j int, n *Neuron) {
				if miss, ok := n.specific.(floatType); ok {
					miss = 0
					for _, w := range p.neuron[i + 1] {
						if m, ok := w.specific.(floatType); ok {
							miss += m * w.axon[j].weight
						}
					}
					miss *= floatType(calcDerivative(float64(n.value), p.activationMode))
					n.specific = miss
				}
				wait <- true
			}(j, v)
		}
		for range p.neuron[i] {
			<- wait
		}
	}
}

// Update weights
func (p *perceptron) calcAxon(input []float64) {
	p.calcMiss(input)
	wait := make(chan bool)
	defer close(wait)
	for _, u := range p.axon {
		for _, v := range u {
			for _, w := range v {
				go func(a *Axon) {
					if n, ok := a.synapse["output"].(*Neuron); ok {
						if miss, ok := n.specific.(floatType); ok {
							a.weight += getSynapseInput(a) * miss * p.rate
						}
					}
					wait <- true
				}(w)
			}
			for range v {
				<- wait
			}
		}
	}
}

// Training
func (p *perceptron) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if len(target) > 0 {
		for count < 1 /*MaxIteration*/ {
			p.calcNeuron(input)
			if loss = p.calcLoss(target[0]); loss <= p.lossLevel || loss <= MinLossLevel {
				break
			}
			//p.calcMiss(input)
			p.calcAxon(input)
			count++
		}
	} else {
		pkg.Log("No target data", true) // !!!
		return -1, 0
	}
	return
}

// Querying
func (p *perceptron) Query(input []float64) (output []float64) {
	p.calcNeuron(input)
	output = make([]float64, p.lenOutput)
	for i, n := range p.neuron[p.lastIndexLayer] {
		output[i] = float64(n.value)
	}
	return
}

// Verifying
func (p *perceptron) Verify(input []float64, target ...[]float64) (loss float64) {
	if len(target) > 0 {
		p.calcNeuron(input)
		loss = p.calcLoss(target[0])
	} else {
		pkg.Log("No target data", true) // !!!
		return -1
	}
	return
}

//
func (p *perceptron) Read(reader io.Reader) {

}

//
func (p *perceptron) Write(writer ...io.Writer) {
	for _, w := range writer {
		switch v := w.(type) {
		case *report:
			p.writeReport(v)
		case jsonType:
			p.writeJSON(string(v))
		/*case xml:
			p.writeXML(v)
		case xml:
			p.writeCSV(v)
		case db:
			p.writeDB(v)*/
		default:
			pkg.Log("This type is missing for write", true) // !!!
			log.Printf("\tWrite: %T %v\n", w, w) // !!!
		}
	}
}

//
func (p *perceptron) setWeight(weight [][][]floatType)  {

}

//
func (p *perceptron) getWeight() [][][]floatType {
	weight := make([][][]floatType, len(p.axon))
	for i, u := range p.axon {
		weight[i] = make([][]floatType, len(p.axon[i]))
		for j, v := range u {
			weight[i][j] = make([]floatType, len(p.axon[i][j]))
			for k, w := range v {
				weight[i][j][k] = w.weight
			}
		}
	}
	return weight
}

//
func (p *perceptron) readJSON(filename string) {
	var test struct{
		Architecture	string			`json:"architecture" xml:"architecture"`
		IsTrain			bool			`json:"isTrain" xml:"isTrain"`
		HiddenLayer		HiddenType		`json:"hiddenLayer" xml:"hiddenLayer"`
		Bias			biasType		`json:"bias" xml:"bias"`
		ActivationMode	uint8			`json:"activationMode" xml:"activationMode"`
		LossMode		uint8			`json:"lossMode" xml:"lossMode"`
		LossLevel		float64			`json:"lossLevel" xml:"lossLevel"`
		Rate			floatType		`json:"rate" xml:"rate"`
		Weights			[][][]floatType	`json:"weights" xml:"weights"`
	}
	t := test
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load settings: ", err)
	}
	err = json.Unmarshal(b, &t)
	if err != nil {
		log.Fatal("Invalid settings format: ", err)
	}

	err = ioutil.WriteFile(filename, b, os.ModePerm)

	fmt.Println(t.Weights)

	if t.Architecture == "perceptron" {

	}

	//fmt.Println("ValueOf ", reflect.ValueOf(t).Field(0)) // perceptron
}

//
func (p *perceptron) writeJSON(filename string) {
	test := struct{
		Architecture	string			`json:"architecture" xml:"architecture"`
		IsTrain			bool			`json:"isTrain" xml:"isTrain"`
		HiddenLayer		HiddenType		`json:"hiddenLayer" xml:"hiddenLayer"`
		Bias			biasType		`json:"bias" xml:"bias"`
		ActivationMode	uint8			`json:"activationMode" xml:"activationMode"`
		LossMode		uint8			`json:"lossMode" xml:"lossMode"`
		LossLevel		float64			`json:"lossLevel" xml:"lossLevel"`
		Rate			floatType		`json:"rate" xml:"rate"`
		Weights			[][][]floatType	`json:"weights" xml:"weights"`
	}{
		Architecture:	"perceptron",
		IsTrain:		p.Architecture.(*NN).isTrain,
		HiddenLayer:	p.hiddenLayer,
		Bias:			p.bias,
		ActivationMode:	p.activationMode,
		LossMode:		p.lossMode,
		LossLevel:		p.lossLevel,
		Rate:			p.rate,
		Weights:		p.getWeight(),
	}
	b, err := json.MarshalIndent(test, "", "\t")
	if err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}

	p.readJSON(filename)

}

// Report of neural network training results in io.Writer
func (p *perceptron) writeReport(report *report) {
	s := "----------------------------------------------\n"
	n := "\n"
	m := "\n\n"
	b := bytes.NewBufferString("Report of Perceptron Neural Network\n\n")

	// Input layer
	if in, ok := report.args[0].([]float64); ok {
		_, _ = fmt.Fprintf(b, "%s0 Input layer size: %d\n%sNeurons:\t", s, p.lenInput, s)
		for _, v := range in {
			_, _ = fmt.Fprintf(b, "  %v", v)
		}
		_, _ = fmt.Fprint(b, m)
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
		_, _ = fmt.Fprintf(b, "%s%d %s size: %d\n%sNeurons:\t", s, i + 1, t, len(p.neuron[i]), s)
		for _, w := range v {
			_, _ = fmt.Fprintf(b, "  %11.8f", w.value)
		}
		_, _ = fmt.Fprint(b, "\nMiss:\t\t")
		for _, w := range v {
			_, _ = fmt.Fprintf(b, "  %11.8f", w.specific)
		}
		_, _ = fmt.Fprint(b, m)
	}

	// Axons: weight
	_, _ = fmt.Fprintf(b, "%sAxons\n%s", s, s)
	for _, u := range p.axon {
		for i, v := range u {
			_, _ = fmt.Fprint(b, i + 1)
			for _, w := range v {
				_, _ = fmt.Fprintf(b, "\t%11.8f", w.weight)
			}
			_, _ = fmt.Fprint(b, n)
		}
		_, _ = fmt.Fprint(b, n)
	}

	// Resume
	if loss, ok := report.args[1].(float64); ok {
		_, _ = fmt.Fprintf(b, "%sTotal error:\t\t%v\n", s, loss)
	}
	if count, ok := report.args[2].(int); ok {
		_, _ = fmt.Fprintf(b, "Number of iteration:\t%v\n", count)
	}

	_, _ = b.WriteTo(report.file)
	_ = report.file.Close()
}