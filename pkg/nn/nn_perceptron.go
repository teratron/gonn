// Perceptron Neural Network
package nn

import (
	"fmt"
	"log"
	"math"
)

type perceptron struct {
	Architecture

	bias			biasType			//
	rate			floatType			//
	modeActivation	modeActivationType	//
	modeLoss		modeLossType		//
	levelLoss		float64				// Minimum (sufficient) level of the average of the error during training
	hiddenLayer		HiddenType			// Array of the number of neurons in each hidden layer

	neuron			[][]*neuron
	axon			[][][]*axon

	lastIndexLayer	int
	lenInput		int
	lenOutput		int
}

type perceptronNeuron struct {
	miss floatType
}

// Returns a new Perceptron neural network instance with the default parameters
func (n *nn) Perceptron() NeuralNetwork {
	n.Architecture = &perceptron{
		Architecture:	n,
		bias:			false,
		rate:			DefaultRate,
		modeActivation:	ModeSIGMOID,
		modeLoss:		ModeMSE,
		levelLoss:		.0001,
		hiddenLayer:	HiddenType{},
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
			Bias(false),
			Rate(DefaultRate),
			ModeActivation(ModeSIGMOID),
			ModeLoss(ModeMSE),
			LevelLoss(.0001),
			HiddenLayer())
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
			p.modeActivation = v
		case modeLossType:
			p.modeLoss = v
		case levelLossType:
			p.levelLoss = float64(v)
		case HiddenType:
			p.hiddenLayer = v
		default:
			Log("This type is missing for Perceptron Neural Network", false) // !!!
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
			return p.modeActivation
		case modeLossType:
			return p.modeLoss
		case levelLossType:
			return levelLossType(p.levelLoss)
		case HiddenType:
			return p.hiddenLayer
		//case *neuron:
			//return nil //&p.neuron
		default:
			Log("This type is missing for Perceptron Neural Network", false) // !!!
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
		tmp              = append(p.hiddenLayer, hiddenType(p.lenOutput))
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
				//fmt.Println("- ", i, j, k, p.axon[i][j][k])
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
	//wait := make(chan bool)
	//defer close(wait)
	for i, v := range p.neuron {
		for j, w := range v {
			fmt.Println("n + ", i, j)
			//fmt.Printf("%T %v\n", w, w)
			go func(n *neuron, i, j int /*ch chan bool*/) {
				//fmt.Println("n - ", i, j)
				n.value = 0
				for _, a := range n.axon {
					n.value += getSynapseInput(a) * a.weight
				}
				n.value = floatType(calcActivation(float64(n.value), p.modeActivation))
				fmt.Println("neuron - ", i, j, n.value)
				//wait <- true
			}(w, i, j/*wait*/)
			//fmt.Println("neuron + ", i, j, w.value)
		}
		_, _ = fmt.Scanln()
		/*for range v {
			//fmt.Println("neuron - ", v)
			<- wait
		}*/
	}
	//fmt.Println("neuron - ", p.neuron)
}

// Calculating the error of the output neuron
func (p *perceptron) calcLoss(target []float64) (loss float64) {
	for i, v := range p.neuron[p.lastIndexLayer] {
		if s, ok := v.specific.(*perceptronNeuron); ok {
			s.miss = floatType(target[i]) - v.value
			switch p.modeLoss {
			default: fallthrough
			case ModeMSE, ModeRMSE:
				loss += math.Pow(float64(s.miss), 2)
			case ModeARCTAN:
				loss += math.Pow(math.Atan(float64(s.miss)), 2)
			}
			s.miss *= floatType(calcDerivative(float64(v.value), p.modeActivation))
		}
	}
	loss /= float64(p.lenOutput)
	if p.modeLoss == ModeRMSE {
		loss = math.Sqrt(loss)
	}
	fmt.Println("loss - ", loss)
	return
}

// Calculating the error of neurons in hidden layers
func (p *perceptron) calcMiss() {
	for i := p.lastIndexLayer - 1; i >= 0; i-- {
		for j, v := range p.neuron[i] {
			go func() {
				if s, ok := v.specific.(*perceptronNeuron); ok {
					s.miss = 0
					for _, w := range p.neuron[i + 1] {
						if m, ok := w.specific.(*perceptronNeuron); ok {
							s.miss += m.miss * w.axon[j].weight
						}
					}
					s.miss *= floatType(calcDerivative(float64(v.value), p.modeActivation))
					fmt.Println("miss - ", i, j, s.miss)
				}
			}()
		}
	}
}

// Update weights
func (p *perceptron) calcAxon() {
	for i, v := range p.axon {
		for j, w := range v {
			for _, a := range w {
				go func() {
					if n, ok := a.synapse["output"].(*neuron); ok {
						if s, ok := n.specific.(*perceptronNeuron); ok {
							a.weight += getSynapseInput(a) * s.miss * p.rate
							fmt.Println("weight - ", i, j, a.weight)
						}
					}
				}()
			}
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