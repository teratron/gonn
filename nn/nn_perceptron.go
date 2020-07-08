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
// args[0] - input data
// args[1] - target data
func (p *perceptron) init(args ...Setter) bool {
	var numAxon int
	numNeuron := 0
	lenHidden := len(p.hiddenLayer)
	lenInput  := len(args[0].(FloatType))
	lenTarget := len(args[1].(FloatType))
	b := 0
	if p.bias {
		b = 1
	}
	lenBias := lenInput + b

	// Определяем количества нейронов и аксонов в матрице
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
		n.neuron     = make([]*neuron, numNeuron)
		n.axon       = make([]*axon, numAxon)
		n.lastNeuron = numNeuron - 1
		n.lastAxon   = numAxon - 1

		for i := 0; i < numNeuron; i++ {
			n.neuron[i] = &neuron{}
			//n.neuron[i].axon[0] = &axon{}
		}
		for i := 0; i < numAxon; i++ {
			n.axon[i] = &axon{
				weight:	 getRand(),
				synapse: map[string]Setter{},
			}
			//n.axon[i].synapse = make(map[string]Setter, 3)


		}


		func(index int) {
			sumAxonLayer, sumNeuronLayer, prevAxonLayer, prevNeuronLayer := 0, 0, 0, 0
			var numNeuronLayer int
			var layer HiddenType
			layer = append(p.hiddenLayer, hiddenType(lenTarget))
			for i, v := range layer {
				if i == 0 {
					numNeuronLayer = lenInput + b
				} else {
					numNeuronLayer = int(layer[i - 1]) + b
				}
				sumAxonLayer += numNeuronLayer * int(v)
				if index < sumAxonLayer {
					for j := 0; j < int(v); j++ {
						delta := index - prevAxonLayer
						if delta < numNeuronLayer * (j + 1) {
							n.axon[index].synapse["output"] = n.neuron[j + sumNeuronLayer]
							delta -= numNeuronLayer * j
							if i == 0 {
								if delta < lenInput {
									if g, ok := args[0].(FloatType); ok {
										n.axon[index].synapse["input"] = floatType(g[delta])
									}
								} else {
									if p.bias {
										n.axon[index].synapse["bias"] = biasType(true)
									} else {
										panic("error") // !!!
									}
								}
							} else {
								n.axon[index].synapse["input"] = n.neuron[prevNeuronLayer + delta]
							}
							fmt.Println("-", n.axon[index])
							break
						}
					}
					break
				}
				prevAxonLayer = sumAxonLayer
				prevNeuronLayer = sumNeuronLayer
				sumNeuronLayer += int(v)
			}
		}(1)


		//
		/*if lenHidden > 0 { // если есть скрытые слои
			var m int
			c, d, e := 0, 0, 0
			for i, v := range p.hiddenLayer { // проходим по скрытым слоям
				for j := 0; j < int(v); j++ { // проходим по нейронам в скрытых слоях
					if i == 0 { // первый скрытый слой
						m  = lenInput
						d += lenBias
						//fmt.Println("--", i, e, c, d)
						for k, in := range args[0].(FloatType) { // проходим по входным нейронам
							n.axon[k + c].synapse["input"]  = floatType(in)
							n.axon[k + c].synapse["output"] = n.neuron[j]
							//fmt.Println("-", c + k, n.axon[k + c])
						}
					} else { // последующие скрытые слои
						m  = int(p.hiddenLayer[i - 1])
						d += m + b
						//fmt.Println("--", i, e, c, d)
						for k := 0; k < m; k++ { // проходим по нейронам предыдущего скрытого слоя
							n.axon[k + c].synapse["input"]  = n.neuron[k + e - m]
							n.axon[k + c].synapse["output"] = n.neuron[j + e]
							//fmt.Println("-", c + k, n.axon[k + c])
						}
					}
					if p.bias {
						n.axon[m + c].synapse["bias"]   = biasType(true)
						n.axon[m + c].synapse["output"] = n.neuron[j + e]
						//fmt.Println("-", c + m, n.axon[c + m])
					}
					n.neuron[j + e].axon = n.axon[c:d]
					//fmt.Println("-+-", n.neuron[j + e].axon)
					c = d
				}
				e += int(v)
			}
			//fmt.Println("-+-+-", e, c, d)
			m = int(p.hiddenLayer[len(p.hiddenLayer) - 1])
			for i := 0; i < lenTarget; i++ { // проходим по выходным нейронам
				for j := 0; j < m; j++ { // проходим по нейронам последнего скрытого слоя
					n.axon[j + c].synapse["input"]  = n.neuron[j + e - m]
					n.axon[j + c].synapse["output"] = n.neuron[i + e]
					//fmt.Println("-", c + j, n.axon[c + j])
				}
				if p.bias {
					n.axon[m + c].synapse["bias"]   = biasType(true)
					n.axon[m + c].synapse["output"] = n.neuron[i + e]
					//fmt.Println("-", c + m, n.axon[c + m])
				}
				d += m + b
				//fmt.Println("-", i, i +e, c, d)
				if d == numAxon {
					n.neuron[i + e].axon = n.axon[c:]
					//fmt.Println("-+-", n.neuron[i + e].axon)
				} else {
					n.neuron[i + e].axon = n.axon[c:d]
					//fmt.Println("-+-", n.neuron[i + e].axon)
				}
				c = d
			}
		} else { // если скрытые слои отсутсвуют
		}*/

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