// Perceptron Neural Network
package nn

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
)

type perceptron struct {
	Architecture

	hiddenLayer		HiddenType	// Array of the number of neurons in each hidden layer
	bias			biasType	// The neuron bias, false or true
	rate			floatType	// Learning coefficient, from 0 to 1
	modeActivation	uint8		// Activation function mode
	modeLoss		uint8		//
	levelLoss		float64		// Minimum (sufficient) level of the average of the error during training

	neuron			[][]*neuron
	axon			[][][]*axon

	lastIndexLayer	int
	lenInput		int
	lenOutput		int
}

type perceptronNeuron struct {
	Miss floatType // Error value
}

// Returns a new Perceptron neural network instance with the default parameters
func (n *nn) Perceptron() NeuralNetwork {
	n.Architecture = &perceptron{
		Architecture:	n,
		hiddenLayer:	HiddenType{},
		bias:			false,
		rate:			floatType(DefaultRate),
		modeActivation:	ModeSIGMOID,
		modeLoss:		ModeMSE,
		levelLoss:		.0001,
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
			Rate(DefaultRate),
			ModeActivation(ModeSIGMOID),
			ModeLoss(ModeMSE),
			LevelLoss(.0001))
	}
}

// Setter
func (p *perceptron) Set(args ...Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		case biasType:
			p.bias = v
		case rateType:
			p.rate = floatType(v)
		case modeActivationType:
			p.modeActivation = uint8(v)
		case modeLossType:
			p.modeLoss = uint8(v)
		case levelLossType:
			p.levelLoss = float64(v)
		case HiddenType:
			p.hiddenLayer = v
		default:
			Log("This type is missing for Perceptron Neural Network", true) // !!!
			log.Printf("\tset: %T %v\n", v, v) // !!!
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (p *perceptron) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		switch args[0].(type) {
		case biasType:
			return p.bias
		case rateType:
			return p.rate
		case modeActivationType:
			return modeActivationType(p.modeActivation)
		case modeLossType:
			return modeLossType(p.modeLoss)
		case levelLossType:
			return levelLossType(p.levelLoss)
		case HiddenType:
			return p.hiddenLayer
		//case *neuron:
			//return nil //&p.neuron
		default:
			Log("This type is missing for Perceptron Neural Network", true) // !!!
			log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
			return nil
		}
	} else {
		return p
	}
}

// Specific neuron
/*func (p *perceptronNeuron) Set(...Setter) {}
*/
func (p *perceptronNeuron) Get(...Getter) GetterSetter {
	return nil
}

// Initialization
func (p *perceptron) init(input []float64, target ...[]float64) bool {
	if len(target) > 0 {
		var tmp HiddenType
		defer func() { tmp = nil }()

		p.lastIndexLayer = len(p.hiddenLayer)
		p.lenInput       = len(input)
		p.lenOutput      = len(target[0])
		tmp              = append(p.hiddenLayer, uint(p.lenOutput))
		layer           := make(HiddenType, p.lastIndexLayer+1)
		lenLayer        := copy(layer, tmp)

		b := 0
		if p.bias { b = 1 }

		p.neuron = make([][]*neuron, lenLayer)
		p.axon   = make([][][]*axon, lenLayer)
		for i, l := range layer {
			p.neuron[i] = make([]*neuron, l)
			p.axon[i]   = make([][]*axon, l)
			for j := 0; j < int(l); j++ {
				if i == 0 {
					p.axon[i][j] = make([]*axon, p.lenInput+b)
				} else {
					p.axon[i][j] = make([]*axon, int(layer[i-1])+b)
				}
			}
		}
		p.initNeuron()
		p.initAxon()
		return true
	} else {
		Log("No target data", true) // !!!
		return false
	}
}

//
func (p *perceptron) initNeuron() {
	for i, v := range p.neuron {
		for j := range v {
			p.neuron[i][j] = &neuron{
				specific: &perceptronNeuron{},
				axon:     p.axon[i][j],
			}
		}
	}
}

//
func (p *perceptron) initAxon() {
	for i, v := range p.axon {
		for j, w := range v {
			for k := range w {
				p.axon[i][j][k] = &axon{
					weight:  getRand(),
					synapse: map[string]Getter{},
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
func (p *perceptron) initSynapse(input []float64) {
	for j, w := range p.axon[0] {
		for k := range w {
			if k < p.lenInput {
				p.axon[0][j][k].synapse["input"] = floatType(input[k])
			}
		}
	}
}

// Calculating the values of neurons in a layers
func (p *perceptron) calcNeuron() {
	wait := make(chan bool)
	defer close(wait)
	for _, v := range p.neuron {
		for _, w := range v {
			go func(n *neuron) {
				n.value = 0
				for _, a := range n.axon {
					n.value += getSynapseInput(a) * a.weight
				}
				n.value = floatType(calcActivation(float64(n.value), p.modeActivation))
				wait <- true
			}(w)
		}
		for range v { <- wait }
	}
}

// Calculating the error of the output neuron
func (p *perceptron) calcLoss(target []float64) (loss float64) {
	for i, v := range p.neuron[p.lastIndexLayer] {
		if s, ok := v.specific.(*perceptronNeuron); ok {
			s.Miss = floatType(target[i]) - v.value
			switch p.modeLoss {
			default: fallthrough
			case ModeMSE, ModeRMSE:
				loss += math.Pow(float64(s.Miss), 2)
			case ModeARCTAN:
				loss += math.Pow(math.Atan(float64(s.Miss)), 2)
			}
			s.Miss *= floatType(calcDerivative(float64(v.value), p.modeActivation))
		}
	}
	loss /= float64(p.lenOutput)
	if p.modeLoss == ModeRMSE {
		loss = math.Sqrt(loss)
	}
	return
}

// Calculating the error of neurons in hidden layers
func (p *perceptron) calcMiss() {
	wait := make(chan bool)
	defer close(wait)
	for i := p.lastIndexLayer - 1; i >= 0; i-- {
		for j, v := range p.neuron[i] {
			go func(n *neuron) {
				if s, ok := n.specific.(*perceptronNeuron); ok {
					s.Miss = 0
					for _, w := range p.neuron[i + 1] {
						if m, ok := w.specific.(*perceptronNeuron); ok {
							s.Miss += m.Miss * w.axon[j].weight
						}
					}
					s.Miss *= floatType(calcDerivative(float64(n.value), p.modeActivation))
				}
				wait <- true
			}(v)
		}
		for range p.neuron[i] { <- wait }
	}
}

// Update weights
func (p *perceptron) calcAxon() {
	wait := make(chan bool)
	defer close(wait)
	for _, u := range p.axon {
		for _, v := range u {
			for _, w := range v {
				go func(a *axon) {
					if n, ok := a.synapse["output"].(*neuron); ok {
						if s, ok := n.specific.(*perceptronNeuron); ok {
							a.weight += getSynapseInput(a) * s.Miss * p.rate
						}
					}
					wait <- true
				}(w)
			}
			for range v { <- wait }
		}
	}
}

// Training
func (p *perceptron) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if len(target) > 0 {
		p.initSynapse(input)
		for count < 1 /*MaxIteration*/ {
			p.calcNeuron()
			if loss = p.calcLoss(target[0]); loss <= p.levelLoss || loss <= MinLevelLoss {
				break
			}
			p.calcMiss()
			p.calcAxon()
			count++
		}
	} else {
		Log("No target data", true) // !!!
		return -1, 0
	}
	return
}

// Querying
func (p *perceptron) Query(input []float64) (output []float64) {
	p.initSynapse(input)
	p.calcNeuron()
	output = make([]float64, p.lenOutput)
	for i, n := range p.neuron[p.lastIndexLayer] {
		output[i] = float64(n.value)
	}
	return
}

// Verifying
func (p *perceptron) Verify(input []float64, target ...[]float64) (loss float64) {
	if len(target) > 0 {
		p.initSynapse(input)
		p.calcNeuron()
		loss = p.calcLoss(target[0])
	} else {
		Log("No target data", true) // !!!
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
		case *os.File:
			continue
		case *reportType:
			//fmt.Printf("report: %T %v\n", v, v.input)
			for _, u := range writer {
				if _, ok := u.(*reportType); !ok {
					if f, ok := u.(*os.File); ok {
						p.writeReport(f, v)
						//fmt.Printf("report: %T %v\n", f, f)
					}
				}
			}
		case jsonType:
			p.writeJSON(v)
		/*case xml:
			p.writeXML(v)
		case db:
			p.writeDB(v)*/
		default:
			Log("This type is missing for write", true) // !!!
			log.Printf("\tWrite: %T %v\n", w, w) // !!!
		}
	}
}

//
func (p *perceptron) writeReport(writer *os.File, report *reportType) {
	var b bytes.Buffer
	b.Write([]byte("Hello "))
	_, _ = fmt.Fprintf(&b, "world!")
	b.WriteTo(writer)
	w := bufio.NewWriter(writer)

	sep  := "----------------------------------------------\n"
	line := "\n\n"

	_, _ = w.WriteString(sep + line)
	err := w.Flush()
	if err != nil {} // !!!
}

//
func (p *perceptron) writeJSON(filename jsonType) {
	//fmt.Println(filename)
	j, err := json.MarshalIndent(p.Architecture.(*nn), "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println(string(j))
}

// Output of neural network training results in io.Writer
func (p *perceptron) Print(writer io.Writer, input []float64, args ...interface{}) {
	sep := func() {
		_, _ = fmt.Fprintln(writer, "----------------------------------------------")
	}

	// Input layer
	sep()
	_, _ = fmt.Fprintln(writer, "0 Input layer size: ", p.lenInput)
	sep()
	_, _ = fmt.Fprint(writer, "Neurons:\t")
	for _, v := range input {
		_, _ = fmt.Fprintf(writer, "  %v", v)
	}
	_, _ = fmt.Fprint(writer, "\n\n")

	// Layers: neuron, miss
	var t string
	for i, v := range p.neuron {
		switch i {
		case p.lastIndexLayer:
			t = "Output layer"
		default:
			t = "Hidden layer"
		}
		sep()
		_, _ = fmt.Fprintf(writer, "%d %s size: %d\n", i + 1, t, len(p.neuron[i]))
		sep()
		_, _ = fmt.Fprint(writer, "Neurons:\t")
		for _, w := range v {
			_, _ = fmt.Fprintf(writer, "  %11.8f", w.value)
		}
		_, _ = fmt.Fprint(writer, "\nMiss:\t\t")
		for _, w := range v {
			_, _ = fmt.Fprintf(writer, "  %11.8f", w.specific.(*perceptronNeuron).Miss)
		}
		_, _ = fmt.Fprint(writer, "\n\n")
	}

	// Axons: weight
	sep()
	_, _ = fmt.Fprintln(writer, "Axons")
	sep()
	for _, u := range p.axon {
		for i, v := range u {
			_, _ = fmt.Fprint(writer, i + 1)
			for _, w := range v {
				_, _ = fmt.Fprintf(writer, "\t%11.8f", w.weight)
			}
			_, _ = fmt.Fprint(writer, "\n")
		}
		_, _ = fmt.Fprint(writer, "\n")
	}

	// Resume
	sep()
	_, _ = fmt.Fprintln(writer, "Number of iteration:\t", args[0])
	_, _ = fmt.Fprintln(writer, "Total error:\t", args[1])
}