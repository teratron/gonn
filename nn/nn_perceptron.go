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
		//neuron:			struct{ error floatType }{error:.2},
	}
	//n.neuron[0].Architecture.neuron = make([]struct{}, 0)
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
	lenBias := lenInput + b

	//
	if lenHidden > 0 {
		for i, v := range p.hiddenLayer {
			numNeuron += int(v)
			if i == 0 {
				numAxon = lenBias * int(v)
			} else {
				numAxon += (int(p.hiddenLayer[i - 1]) + b) * int(v)
			}
		}
		numAxon += (int(p.hiddenLayer[lenHidden - 1]) + b) * lenTarget
	} else {
		numAxon = lenBias * lenTarget
	}
	numNeuron += lenTarget

	//
	if n, ok := p.Architecture.(*nn); ok {
		n.neuron     = make([]neuron, numNeuron)
		n.axon       = make([]axon, numAxon)
		n.lastNeuron = numNeuron - 1
		n.lastAxon   = numAxon - 1

		for i := 0; i < numNeuron; i++ {
			n.neuron[i] = neuron{}
			//n.neuron[i].axon[0] = &axon{}
		}
		//fmt.Println("+", &n.axon[0], n.axon)
		for i := 0; i < numAxon; i++ {
			//fmt.Println("+", &n.axon[i])
			n.axon[i] = axon{
				weight:	 getRand(),
				synapse: map[string]Setter{},
			}
			//n.axon[i].synapse = make(map[string]Setter, 3)
		}
		//fmt.Println("+", &n.axon[0], n.axon)
		/*for i := 0; i < numNeuron; i++ {
			if i >= 0 && i < 0 + int(p.hiddenLayer[0]) { // 0-2
				if i == 0 {
					n.neuron[0].axon = n.axon[:lenInput]
					fmt.Println("-", i, p.hiddenLayer[0], n.neuron[i].axon, len(n.neuron[i].axon))
				} else {
					n.neuron[i].axon = n.axon[i * lenInput:lenInput*(i + 1)]
					//n.neuron[i].axon = n.axon[i * lenInput:lenInput*(i + 1)]
					fmt.Println("-", i, p.hiddenLayer[0], n.neuron[i].axon)
				}
				//for j := 0 ; j ; j++ {
				//}
			}
			//fmt.Println("-", int(p.hiddenLayer[0]), int(p.hiddenLayer[0] + p.hiddenLayer[1]))
			if i >= 0 + int(p.hiddenLayer[0]) && i < int(p.hiddenLayer[0] + p.hiddenLayer[1]) { // 3-4
				if i < numNeuron - 1 {
					n.neuron[i].axon = n.axon[i * int(p.hiddenLayer[1]):int(p.hiddenLayer[0])*(i + 1)]
					//fmt.Println("-", p.hiddenLayer[0], int(p.hiddenLayer[0])*(i + 1))
				} else {
					n.neuron[i].axon = n.axon[i * int(p.hiddenLayer[1]) :]
				}
				fmt.Println("-", i, p.hiddenLayer[1], n.neuron[i].axon)
			}
			if i >= int(p.hiddenLayer[0] + p.hiddenLayer[1]) && i < int(p.hiddenLayer[0] + p.hiddenLayer[1]) + lenTarget { // 5-5
			}
		}*/


		/*for i := 0; i < numAxon; i++ {
			switch {
			case i < lenInput:

			}
		}
		//
		if lenHidden > 0 {
			for i := 0; i < numNeuron; i++ {

			}
		} else {
		}*/

		//
		if lenHidden > 0 { // если есть скрытые слои
			c, d, e := 0, 0, 0
			for i, v := range p.hiddenLayer { // проходим по скрытым слоям
				for j := 0; j < int(v); j++ { // проходим по нейронам в скрытых слоях
					if i == 0 { // первый скрытый слой
						d += lenBias
						fmt.Println("--", i, e, c, d)
						for k, in := range data[0] { // проходим по входным нейронам
							n.axon[c + k].synapse["input"]  = floatType(in)
							n.axon[c + k].synapse["output"] = &n.neuron[j]
							fmt.Println("-", c + k, n.axon[c + k])
							//fmt.Println("-", m + k, n.axon[m + k].synapse["input"], n.axon[m + k].synapse["output"])
						}
						if p.bias {
							n.axon[c + lenInput].synapse["bias"]   = biasType(true)
							n.axon[c + lenInput].synapse["output"] = &n.neuron[j]
							fmt.Println("-", c + lenInput, n.axon[c + lenInput])
							//fmt.Println("-", m + lenInput, n.axon[m + lenInput].synapse["bias"], n.axon[m + lenInput].synapse["output"])
						}
					} else {
						d += int(p.hiddenLayer[i - 1]) + b
						fmt.Println("--", i, e, c, d)
						for k := 0; k < int(p.hiddenLayer[i - 1]); k++ { // проходим по предыдущим нейронам
							n.axon[c + k].synapse["input"]  = &n.neuron[j]
							n.axon[c + k].synapse["output"] = &n.neuron[j]
							fmt.Println("-", c + k, n.axon[c + k])
						}
						if p.bias {
						}
					}
					n.neuron[j + e].axon = n.axon[c : d]
					c = d
				}
				e += int(v)
			}
		} else {
		}

		// Fills all weights with random numbers
		//n.setRandWeight()
	}
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