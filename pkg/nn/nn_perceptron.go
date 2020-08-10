// Perceptron Neural Network
package nn

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

var _ NeuralNetwork = (*perceptron)(nil)
//var _ PerceptronNN = (*perceptron)(nil)

/*type PerceptronNN interface {
	HiddenLayer() []uint
	Bias() bool
	ActivationMode() uint8
	LossMode() uint8
	LossLevel() float64
	Rate() float32
}*/


type perceptron struct {
	architecture		NeuralNetwork

	Parameter struct{
		// Array of the number of neurons in each hidden layer
		HiddenLayer		HiddenType		`json:"hiddenLayer" xml:"hiddenLayer"`

		// The neuron bias, false or true
		Bias			biasType		`json:"bias" xml:"bias"`

		// Activation function mode
		ActivationMode	uint8			`json:"activationMode" xml:"activationMode"`

		// The mode of calculation of the total error
		LossMode		uint8			`json:"lossMode" xml:"lossMode"`

		// Minimum (sufficient) level of the average of the error during training
		LossLevel		float64			`json:"lossLevel" xml:"lossLevel"`

		// Learning coefficient, from 0 to 1
		Rate			floatType		`json:"rate" xml:"rate"`

		//
		Weights			[][][]floatType	`json:"weights" xml:"weights"`
	}									`json:"perceptron" xml:"perceptron"`

	// Matrix
	neuron			[][]*Neuron
	axon			[][][]*Axon

	lastIndexLayer	int
	lenInput		int
	lenOutput		int
}

func Perceptron() *perceptron {
	return &perceptron{}
}

// Returns a new Perceptron neural network instance with the default parameters
func (n *nn) perceptron() NeuralNetwork {
	n.Architecture = &perceptron{
		architecture: n,
	}
	if p, ok := n.Architecture.(*perceptron); ok {
		p.Parameter.HiddenLayer		= HiddenType{9, 2}
		p.Parameter.Bias 			= false
		p.Parameter.ActivationMode	= ModeSIGMOID
		p.Parameter.LossMode		= ModeMSE
		p.Parameter.LossLevel		= .0001
		p.Parameter.Rate			= floatType(DefaultRate)
		p.Parameter.Weights			= nil
	}
	return n
}

// HiddenLayer
/*func (p *perceptron) HiddenLayer() []uint {
	return p.Parameter.HiddenLayer
}

// Bias
func (p *perceptron) Bias() bool {
	return bool(p.Parameter.Bias)
}

// ActivationMode
func (p *perceptron) ActivationMode() uint8 {
	return p.Parameter.ActivationMode
}

// LossMode
func (p *perceptron) LossMode() uint8 {
	return p.Parameter.LossMode
}

// LossLevel
func (p *perceptron) LossLevel() float64 {
	return p.Parameter.LossLevel
}

// Rate
func (p *perceptron) Rate() float32 {
	return float32(p.Parameter.Rate)
}*/

// Setter
func (p *perceptron) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		case HiddenType:
			p.Parameter.HiddenLayer = v
		case biasType:
			p.Parameter.Bias = v
		case activationModeType:
			p.Parameter.ActivationMode = uint8(v)
		case lossModeType:
			p.Parameter.LossMode = uint8(v)
		case lossLevelType:
			p.Parameter.LossLevel = float64(v)
		case rateType:
			p.Parameter.Rate = floatType(v)
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
			return p.Parameter.HiddenLayer
		case biasType:
			return p.Parameter.Bias
		case activationModeType:
			return activationModeType(p.Parameter.ActivationMode)
		case lossModeType:
			return lossModeType(p.Parameter.LossMode)
		case lossLevelType:
			return lossLevelType(p.Parameter.LossLevel)
		case rateType:
			return p.Parameter.Rate
		default:
			pkg.Log("This type is missing for Perceptron Neural Network", true) // !!!
			log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
			return nil
		}
	} else {
		if a, ok := p.architecture.(NeuralNetwork); ok {
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

		p.lastIndexLayer = len(p.Parameter.HiddenLayer)
		p.lenInput       = lenInput
		p.lenOutput      = lenTarget[0].(int)
		tmp              = append(p.Parameter.HiddenLayer, uint(p.lenOutput))
		layer           := make(HiddenType, p.lastIndexLayer + 1)
		lenLayer        := copy(layer, tmp)

		b := 0
		if p.Parameter.Bias {
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
		if n, ok := p.Get().(*nn); ok && !n.IsTrain {
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
				n.value = floatType(calcActivation(float64(n.value), p.Parameter.ActivationMode))
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
			switch p.Parameter.LossMode {
			default: fallthrough
			case ModeMSE, ModeRMSE:
				loss += math.Pow(float64(miss), 2)
			case ModeARCTAN:
				loss += math.Pow(math.Atan(float64(miss)), 2)
			}
			miss *= floatType(calcDerivative(float64(v.value), p.Parameter.ActivationMode))
			v.specific = miss
		}
	}
	loss /= float64(p.lenOutput)
	if p.Parameter.LossMode == ModeRMSE {
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
					miss *= floatType(calcDerivative(float64(n.value), p.Parameter.ActivationMode))
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
							a.weight += getSynapseInput(a) * miss * p.Parameter.Rate
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
			if loss = p.calcLoss(target[0]); loss <= p.Parameter.LossLevel || loss <= MinLossLevel {
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

// setWeight
func (p *perceptron) setWeight(weight [][][]floatType)  {
	for i, u := range weight {
		for j, v := range u {
			for k, w := range v {
				p.axon[i][j][k].weight = w
			}
		}
	}
}

// getWeight
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

// Read
func (p *perceptron) Read(reader pkg.Reader) {
	switch r := reader.(type) {
	case jsonType:
		p.readJSON(string(r))
	/*case xml:
		p.readXML(v)
	case xml:
		p.readCSV(v)
	case db:
		p.readDB(v)*/
	default:
		pkg.Log("This type is missing for write", true) // !!!
		log.Printf("\tWrite: %T %v\n", r, r) // !!!
	}
}

// Write
func (p *perceptron) Write(writer ...pkg.Writer) {
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

// readJSON
func (p *perceptron) readJSON(filename string) {
	/*var test struct{
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
	t := test*/
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load settings: ", err)
	}

	/*var v interface{}
	err = json.Unmarshal(b, &v)
	data := v.(map[string]interface{})
	fmt.Printf("%T - %v", data, data)*/



	//fmt.Println(string(b))
	err = json.Unmarshal(b, &p.Parameter)
	if err != nil {
		log.Fatal("Invalid settings format: ", err)
	}
	fmt.Println(p.Parameter)



	//err = ioutil.WriteFile(filename, b, os.ModePerm)
	//fmt.Println(t.Weights)
	/*if t.Architecture == "perceptron" {
	}*/
	//fmt.Println("ValueOf ", reflect.ValueOf(t).Field(0)) // perceptron
}

// writeJSON
func (p *perceptron) writeJSON(filename string) {
	/*t1 := test1{Name: "1"}
	///t2 := test1{Name2: "2"}
	t := test0{
		//Type: t1,
		Architecture: map[string]Tester{
			"perceptron": t1,
		},
	}*/

	/*v := map[string]interface{}{
		"architecture": p,
	}*/


	b, err := json.MarshalIndent(p, "", "\t")
	if err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}
}

// writeReport report of neural network training results in io.Writer
func (p *perceptron) writeReport(report *report) {
	s := "----------------------------------------------\n"
	n := "\n"
	m := "\n\n"
	b := bytes.NewBufferString("Report of Perceptron Neural Network\n\n")

	printFormat := func(format string, a ...interface{}) {
		_, err := fmt.Fprintf(b, format, a...)
		if err != nil {
			log.Fatal("")
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
		printFormat("%s%d %s size: %d\n%sNeurons:\t", s, i + 1, t, len(p.neuron[i]), s)
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
	printFormat("%sAxons\n%s", s, s)
	for _, u := range p.axon {
		for i, v := range u {
			printFormat("%d", i + 1)
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

	_, err := b.WriteTo(report.file)
	err = report.file.Close()
	if err != nil {
		log.Fatal("")
	}
}