// Perceptron Neural Network
package nn

import (
	"fmt"
	"log"
)

/*type Perceptron interface {
	Perceptron() NeuralNetwork
}*/

type perceptron struct {
	Architecture
	Processor

	bias			biasType			//
	rate			rateType			//
	modeActivation	modeActivationType	//

	modeLoss		modeLossType		//
	levelLoss		levelLossType		// Minimum (sufficient) level of the average of the error during training

	hiddenLayer		HiddenType			// Array of the number of neurons in each hidden layer

	upperRange		floatType			// Range, Bound, Limit, Scope
	lowerRange		floatType

	neuron struct {
		error		floatType
	}
}

// Initializing Perceptron Neural Network
func (n *nn) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{
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
	//fmt.Println(n.architecture)
	//n.neuron[0].architecture =
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
			Activation(ModeSIGMOID),
			Loss(ModeMSE),
			LevelLoss(.0001),
			HiddenLayer())
	}
}

// Setter
func (p *perceptron) Set(set ...Setter) {
	switch v := set[0].(type) {
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
		Log("This type of variable is missing for Perceptron Neural Network", false)
		log.Printf("\tset: %T %v\n", v, v)
	}
}

// Getter
func (p *perceptron) Get(set ...Setter) Getter {
	switch set[0].(type) {
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
		Log("This type of variable is missing for Perceptron Neural Network", false)
		log.Printf("\tget: %T %v\n", set[0], set[0])
		return nil
	}
}

// Init
// data[0] - input data
// data[1] - target data
// ... - any data
func (p *perceptron) init(data ...[]float64) bool {
//func (p *perceptron) init(input, target []float64) bool {
	var numAxon int
	numNeuron := 0
	lenHidden := len(p.hiddenLayer)
	lenInput  := len(data[0])
	lenTarget := len(data[1])
	b := 0
	if p.bias {
		b = 1
	}

	//
	if lenHidden > 0 {
		for i, v := range p.hiddenLayer {
			numNeuron += int(v)
			if i == 0 {
				numAxon = (lenInput + b) * int(v)
			} else {
				numAxon += (int(p.hiddenLayer[i - 1]) + b) * int(v)
			}
		}
		numAxon += (int(p.hiddenLayer[lenHidden - 1]) + b) * lenTarget
	} else {
		numAxon = (lenInput + b) * lenTarget
	}
	numNeuron += lenTarget

	if n, ok := p.Architecture.(*nn); ok {
		n.neuron     = make([]*neuron, numNeuron)
		n.axon       = make([]*axon, numAxon)
		n.lastNeuron = numNeuron - 1
		n.lastAxon   = numAxon - 1

		for i := 0; i < numNeuron; i++ {
			n.neuron[i] = &neuron{}
		}
		for i := 0; i < numAxon; i++ {
			n.axon[i] = &axon{
				weight:		getRand(),
				synapse:	map[string]Setter{},
			}
			//n.axon[i].synapse["string"] = n.neuron[i]
			//fmt.Printf("%T %v\n", n.axon[0].synapse, n.axon[0].synapse)
		}

		//
		if lenHidden > 0 {
			for i, v := range p.hiddenLayer { // проходим по скрытым слоям
				for j := 0; j < int(v); j++ { // проходим по нейронам в скрытых слоях
					if i == 0 {
						m := j * (lenInput + b)
						for k, in := range data[0] { // проходим по входным нейронам
							n.axon[m + k].synapse["input"]  = floatType(in)
							n.axon[m + k].synapse["output"] = n.neuron[j]
							fmt.Println("-", m + k, n.axon[m + k].synapse["input"], n.axon[m + k].synapse["output"])
						}
						if p.bias {
							n.axon[m + lenInput].synapse["bias"]   = biasType(true)
							n.axon[m + lenInput].synapse["output"] = n.neuron[j]
						}
						fmt.Println("-", m + lenInput, n.axon[m + lenInput].synapse["bias"], n.axon[m + lenInput].synapse["output"])
					} else {
						//n.axon[j].synapse["input"] = n.neuron[i]
					}
				}

			}
		} else {

		}

		// Fills all weights with random numbers
		//n.setRandWeight()
		//fmt.Printf("************************ %T %v\n", n, n)

		//n.axon[0].synapse =
	}
	//fmt.Println(numNeuron, numAxon)

	return true
}

// Train
/*func (p *perceptron) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (p *perceptron) Query(input []float64) []float64 {
	panic("implement me")
}*/

/*func (p *perceptron) initHidden() {
}*/