// Perceptron Neural Network
package nn

import (
	"log"
	"math"
)

type perceptron struct {
	Architecture
	Processor

	bias			biasType			//
	rate			rateType			//
	modeActivation	modeActivationType	//
	modeLoss		modeLossType		//
	levelLoss		levelLossType		// Minimum (sufficient) level of the average of the error during training
	hiddenLayer		HiddenType			// Array of the number of neurons in each hidden layer
	lowerRange		lowerRangeType		// Range, Bound, Limit, Scope
	upperRange		upperRangeType

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
		lowerRange:		0,
		upperRange:		1,
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
			HiddenLayer(),
			LowerRange(0),
			UpperRange(1))
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
			p.levelLoss = v
		case HiddenType:
			p.hiddenLayer = v
		case lowerRangeType:
			p.lowerRange = v
		case upperRangeType:
			p.upperRange = v
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
			return p.levelLoss
		case HiddenType:
			return p.hiddenLayer
		case lowerRangeType:
			return p.lowerRange
		case upperRangeType:
			return p.upperRange
		default:
			Log("This type is missing for Perceptron Neural Network", false) // !!!
			log.Printf("\tget: %T %v\n", args[0], args[0])                   // !!!
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
func (p *perceptron) calc(args ...GetterSetter) Getter {
	if len(args) > 0 {
		for _, a := range args {
			switch v := a.(type) {
			case *neuron:
				p.calcNeuron()
			case *axon:
				p.calcAxon()
			case lossType:
				return p.calcLoss(v)
			default:
				Log("This type is missing for Perceptron Neural Network", true) // !!!
				log.Printf("\tcalc: %T %v\n", args[0], args[0]) // !!!
			}
		}
	} else {
		Log("Empty calc()", true)
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
}

// Loosing
func (p *perceptron) Loss(target []float64) float64 {
	return float64(p.calcLoss(target))
}

func (p *perceptron) calcLoss(target lossType) (loss floatType) {
	for i, v := range p.neuron[len(p.neuron) - 1] {
		if s, ok := v.specific.(*perceptronNeuron); ok {
			s.miss = floatType(target[i]) - v.value
			switch p.modeLoss {
			default: fallthrough
			case ModeMSE, ModeRMSE:
				loss += floatType(math.Pow(float64(s.miss), 2))
			case ModeARCTAN:
				loss += floatType(math.Pow(math.Atan(float64(s.miss)), 2))
			}
			s.miss *= floatType(calcDerivative(float64(v.value), p.modeActivation))
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

// Training
func (p *perceptron) Train(data ...[]float64) (loss float64, count int) {
	for count < int(MaxIteration) {
		l, ok := p.calc(Neuron(), Loss(data[1])).(levelLossType)
		if ok && l <= p.levelLoss || l <= MinLevelLoss {
			return float64(l), count
		}
		p.calc(/*Miss(),*/ Axon())
		count++
	}
	return
}

// Querying
/*func (p *perceptron) Query(input []float64) []float64 {
	panic("implement me")
}*/