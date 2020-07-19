// Perceptron Neural Network
package nn

import (
	"log"
	"math"
)

type perceptron struct {
	Architecture

	bias			biasType			//
	rate			rateType			//
	modeActivation	modeActivationType	//
	modeLoss		modeLossType		//
	levelLoss		float64				// Minimum (sufficient) level of the average of the error during training
	hiddenLayer		HiddenType			// Array of the number of neurons in each hidden layer

	neuron			[][]*neuron
	axon			[][][]*axon
}

type perceptronNeuron struct {
	miss			floatType
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
			p.rate = v
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
func (p *perceptronNeuron) Set(...Setter) {}

func (p *perceptronNeuron) Get(...Getter) GetterSetter {
	return nil
}

// Initialization
// args[0] - input data
// args[1] - target data
func (p *perceptron) init(args ...[]float64) bool {
	var tmp HiddenType
	defer func() { tmp = nil }()

	lenHidden := len(p.hiddenLayer)
	layer     := make(HiddenType, lenHidden + 1)
	lenInput  := len(args[0])
	tmp        = append(p.hiddenLayer, hiddenType(len(args[1])))
	lenLayer  := copy(layer, tmp)

	b := 0
	if p.bias { b = 1 }

	p.neuron = make([][]*neuron, lenLayer)
	p.axon   = make([][][]*axon, lenLayer)
	for i, l := range layer {
		p.neuron[i] = make([]*neuron, l)
		p.axon[i]   = make([][]*axon, l)
		for j := 0; j < int(l); j++ {
			if i == 0 {
				p.axon[i][j] = make([]*axon, lenInput + b)
			} else {
				p.axon[i][j] = make([]*axon, int(layer[i - 1]) + b)
			}
		}
	}
	p.initNeuron()
	p.initAxon(args[0])

	return true
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
func (p *perceptron) initAxon(input []float64) {
	for i, v := range p.axon {
		for j, w := range v {
			for k := range w {
				p.axon[i][j][k] = &axon{
					weight:  getRand(),
					synapse: map[string]Getter{},
				}
				if i == 0 {
					if k < len(input) {
						p.axon[i][j][k].synapse["input"] = floatType(input[k])
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

// Calculating the values of neurons in a layers
func (p *perceptron) calcNeuron() {
	var n floatType
	for _, v := range p.neuron {
		for _, w := range v {
			go func() {
				w.value = 0
				for _, a := range w.axon {
					switch s := a.synapse["input"].(type) {
					case floatType:
						n = s
					case biasType:
						if s { n = 1 }
					case *neuron:
						n = s.value
					default:
						panic("error!!!") // !!!
					}
					w.value += n * a.weight
				}
				w.value = floatType(calcActivation(float64(w.value), p.modeActivation))
				//fmt.Println("- ",w.value)
			}()
		}
	}
}

// Loosing
func (p *perceptron) Loss(target []float64) (loss float64) {
	return p.calcLoss(target)
}

// Calculating the error of the output neuron
func (p *perceptron) calcLoss(target []float64) (loss float64) {
	n := len(p.neuron) - 1
	for i, v := range p.neuron[n] {
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
	loss /= float64(len(p.neuron[n]))
	if p.modeLoss == ModeRMSE {
		loss = math.Sqrt(loss)
	}
	return
}

// Calculating the error of neurons in hidden layers
func (p *perceptron) calcMiss() {
	for i := len(p.neuron) - 2; i >= 0; i-- {
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
				}
			}()
		}
	}
}

// Update weights
func (p *perceptron) calcAxon() {
/*var n floatType
	for i, v := range p.axon {
		for j, w := range v {
			for k, u := range w {
				//u.weight += u.synapse["output"].(*neuron).
				switch s := u.synapse["input"].(type) {
				case floatType:
					n = s
				case biasType:
					if s { n = 1 }
				case *neuron:
					n = s.value
				default:
					panic("error!!!") // !!!
				}
			}
		}
	}


	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j, v := range m.Layer[i].Error {
			for k, z := range m.Layer[n].Neuron {
				m.Synapse[n].Weight[k][j] += v * z * m.Rate
			}
		}
	}*/
}

// Training
func (p *perceptron) Train(data ...[]float64) (loss float64, count int) {
	//for count < int(MaxIteration) {
	for count < 10 {
		p.calcNeuron()
		if loss = p.calcLoss(data[1]); loss <= p.levelLoss || loss <= MinLevelLoss { break }
		p.calcMiss()
		p.calcAxon()
		count++
	}
	return
}

// Querying
func (p *perceptron) Query(input []float64) (output []float64) {
	return
}