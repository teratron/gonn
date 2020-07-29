// Perceptron Neural Network
package nn

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
)

type perceptron struct {
	Architecture `json:"-"`

	hiddenLayer    HiddenType // Array of the number of neurons in each hidden layer
	bias           biasType   // The neuron bias, false or true
	rate           floatType  // Learning coefficient, from 0 to 1
	modeActivation uint8      // Activation function mode
	modeLoss       uint8      //
	levelLoss      float64    // Minimum (sufficient) level of the average of the error during training

	neuron			[][]*neuron
	axon			[][][]*axon

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
		rate:           floatType(DefaultRate),
		modeActivation: ModeSIGMOID,
		modeLoss:       ModeMSE,
		levelLoss:      .0001,
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

// Initialization
func (p *perceptron) init(input []float64, target ...[]float64) bool {
	if len(target) > 0 {
		var tmp HiddenType
		defer func() {
			tmp = nil
		}()

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
					p.axon[i][j] = make([]*axon, p.lenInput + b)
				} else {
					p.axon[i][j] = make([]*axon, int(layer[i - 1]) + b)
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
				p.axon[i][j][k] = &axon{
					weight:  .5,//getRand(),
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
			go func(n *neuron) {
				n.value = 0
				for _, a := range n.axon {
					n.value += getSynapseInput(a) * a.weight
				}
				n.value = floatType(calcActivation(float64(n.value), p.modeActivation))
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
			switch p.modeLoss {
			default: fallthrough
			case ModeMSE, ModeRMSE:
				loss += math.Pow(float64(miss), 2)
			case ModeARCTAN:
				loss += math.Pow(math.Atan(float64(miss)), 2)
			}
			miss *= floatType(calcDerivative(float64(v.value), p.modeActivation))
			v.specific = miss
		}
	}
	loss /= float64(p.lenOutput)
	if p.modeLoss == ModeRMSE {
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
			go func(j int, n *neuron) {
				if miss, ok := n.specific.(floatType); ok {
					miss = 0
					for _, w := range p.neuron[i + 1] {
						if m, ok := w.specific.(floatType); ok {
							miss += m * w.axon[j].weight
						}
					}
					miss *= floatType(calcDerivative(float64(n.value), p.modeActivation))
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
				go func(a *axon) {
					if n, ok := a.synapse["output"].(*neuron); ok {
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
			if loss = p.calcLoss(target[0]); loss <= p.levelLoss || loss <= MinLevelLoss {
				break
			}
			//p.calcMiss(input)
			p.calcAxon(input)
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
		case *report:
			p.writeReport(v)
		case jsonType:
			p.writeJSON(v)
		/*case xml:
			p.writeXML(v)
		case xml:
			p.writeCSV(v)
		case db:
			p.writeDB(v)*/
		default:
			Log("This type is missing for write", true) // !!!
			log.Printf("\tWrite: %T %v\n", w, w) // !!!
		}
	}
}

//
func (p *perceptron) writeJSON(filename jsonType) {
	//fmt.Println(filename)
	j, err := json.MarshalIndent(p.Architecture.(*NN), "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println(string(j))

	j, err = json.MarshalIndent(p, "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println(string(j))

	j, err = json.MarshalIndent(p.hiddenLayer, "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println(string(j))

	j, err = json.MarshalIndent(p.bias, "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println(string(j))

	j, err = json.MarshalIndent(p.axon[0][0][0].synapse, "", "\t")
	if err != nil { panic("!!!") }
	fmt.Println(string(j))
}

// Report of neural network training results in io.Writer
func (p *perceptron) writeReport(report *report) {
	s := "----------------------------------------------\n"
	n := "\n"
	m := "\n\n"
	b := bytes.NewBufferString("Report of Perceptron Neural Network" + m)

	// Input layer
	_, _ = fmt.Fprintf(b, "%s0 Input layer size: %d\n%sNeurons:\t", s, p.lenInput, s)
	for _, v := range report.input {
		_, _ = fmt.Fprintf(b, "  %v", v)
	}
	_, _ = fmt.Fprint(b, m)

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
	_, _ = fmt.Fprintf(b, "%sTotal error:\t\t%v\n", s,  report.args[0])
	_, _ = fmt.Fprintf(b, "Number of iteration:\t%v\n", report.args[1])

	_, _ = b.WriteTo(report.writer)
}