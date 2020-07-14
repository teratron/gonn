// Perceptron Neural Network
package nn

import (
	"log"
	"math"
)

type Perceptron interface {
	Perceptron() NeuralNetwork
}

type perceptron struct {
	Architecture
	Processor

	bias			biasType			//
	rate			rateType			//
	modeActivation	modeActivationType	//

	modeLoss		modeLossType		//
	levelLoss		levelLossType		// Minimum (sufficient) level of the average of the error during training

	hiddenLayer		HiddenType			// Array of the number of neurons in each hidden layer

	neuron			[][]*neuron
	axon			[][][]*axon

	upperRange		floatType			// Range, Bound, Limit, Scope
	lowerRange		floatType
}

type perceptronNeuron struct {
	error			floatType
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
		upperRange:		1,
		lowerRange:		0,
	}
	return n
}

func (p *perceptron) architecture() *perceptron {
	return p
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
			Activation(ModeSIGMOID),
			ModeLoss(ModeMSE),
			LevelLoss(.0001),
			HiddenLayer())
	}
}

// Setter
func (p *perceptron) Set(args ...Setter) {
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
		p.levelLoss = v
	case HiddenType:
		p.hiddenLayer = v
	default:
		Log("This type is missing for Perceptron Neural Network", false) // !!!
		log.Printf("\tset: %T %v\n", v, v) // !!!
	}
}

// Getter
func (p *perceptron) Get(args ...Setter) Getter {
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
		return p.levelLoss
	case HiddenType:
		return p.hiddenLayer
	default:
		if len(args) == 0 { return p }
		Log("This type is missing for Perceptron Neural Network", false) // !!!
		log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
		return nil
	}
}

// Initialization
// args[0] - input data
// args[1] - target data
func (p *perceptron) init(args ...Setter) bool {
	var tmp HiddenType
	defer func() { tmp = nil }()

	lenHidden := len(p.hiddenLayer)
	layer     := make(HiddenType, lenHidden + 1)
	lenInput  := len(args[0].(FloatType))
	tmp        = append(p.hiddenLayer, hiddenType(len(args[1].(FloatType))))
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
	p.initAxon(args[0].(FloatType))

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
func (p *perceptron) initAxon(input FloatType) {
	for i, v := range p.axon {
		for j, w := range v {
			for k := range w {
				p.axon[i][j][k] = &axon{
					weight:  getRand(),
					synapse: map[string]GetterSetter{},
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

// Calculating
func (p *perceptron) calc(args ...Initer) Getter {
	switch args[0].(type) {
	case *neuron:
		p.calcNeuron()
	case *axon:
		p.calcAxon()
	default:
		Log("This type is missing for Perceptron Neural Network", false) // !!!
		log.Printf("\tcalc: %T %v\n", args[0], args[0]) // !!!
	}
	return nil
}

// Function for calculating the values of neurons in a layers
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

//
func (p *perceptron) calcAxon() {
	//fmt.Println("2 ###################################")
}

// Loosing
func (p *perceptron) Loss(target []float64) float64 {
	return float64(p.calcLoss(target))
}

func (p *perceptron) calcLoss(target FloatType) (loss floatType) {
	for i, v := range p.neuron[len(p.neuron) - 1] {
		if s, ok := v.specific.(*perceptronNeuron); ok {
			s.error = floatType(target[i]) - v.value

			switch p.modeLoss {
			default: fallthrough
			case ModeMSE, ModeRMSE:
				loss += floatType(math.Pow(float64(s.error), 2))
			case ModeARCTAN:
				loss += floatType(math.Pow(math.Atan(float64(s.error)), 2))
			}
			s.error *= floatType(calcDerivative(float64(v.value), p.modeActivation))
		}
	}
	loss /= floatType(len(p.neuron[len(p.neuron) - 1]))

	switch p.modeLoss {
	default: fallthrough
	case ModeMSE, ModeARCTAN:
		return loss
	case ModeRMSE:
		return floatType(math.Sqrt(float64(loss)))
	}
}

//
/*func (p *perceptronNeuron) calc(args ...Initer) {
}*/

func (p *perceptronNeuron) Set(args ...Setter) {
}

func (p *perceptronNeuron) Get(args ...Setter) Getter {
	return nil
}

// Training
/*func (p *perceptron) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (p *perceptron) Query(input []float64) []float64 {
	panic("implement me")
}*/

/*func (p *perceptron) initHidden() {
}*/