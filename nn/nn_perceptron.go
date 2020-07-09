// Perceptron Neural Network
package nn

import (
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

	neuron			[][]*neuron
	axon			[][][]*axon

	upperRange		floatType			// Range, Bound, Limit, Scope
	lowerRange		floatType
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
		Log("This type of variable is missing for Perceptron Neural Network", false) // !!!
		log.Printf("\tset: %T %v\n", v, v) // !!!
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
		Log("This type of variable is missing for Perceptron Neural Network", false) // !!!
		log.Printf("\tget: %T %v\n", set[0], set[0]) // !!!
		return nil
	}
}

// Init
// args[0] - input data
// args[1] - target data
func (p *perceptron) init(args ...Setter) bool {
	var tmp HiddenType
	//var numAxon int
	//numNeuron := 0
	lenHidden := len(p.hiddenLayer)
	layer     := make(HiddenType, lenHidden + 1)
	lenInput  := len(args[0].(FloatType))
	lenTarget := len(args[1].(FloatType))
	tmp        = append(p.hiddenLayer, hiddenType(lenTarget))
	lenLayer  := copy(layer, tmp)

	defer func() { tmp = nil }()

	b := 0
	if p.bias { b = 1 }
	//lenBias := lenInput + b

	// Определяем количества нейронов и аксонов в матрице
	/*if lenHidden > 0 {
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
	numNeuron += lenTarget*/

	//
	p.neuron = make([][]*neuron, lenLayer)
	p.axon   = make([][][]*axon, lenLayer)

	for i, l := range layer {
		p.neuron[i] = make([]*neuron, l)
		p.axon[i]   = make([][]*axon, l)
		//fmt.Println(i, l, p.neuron[i], p.axon[i])
		for j := 0; j < int(l); j++ {
			//fmt.Println(i, j)
			if i == 0 {
				p.axon[i][j] = make([]*axon, lenInput + b)
				//fmt.Println("- ",i, j, lenInput + b)
			} else {
				p.axon[i][j] = make([]*axon, int(layer[i - 1]) + b)
				//fmt.Println("- ",i, j, int(layer[i - 1]) + b)
			}

		}
	}

		/*for i := 0; i < numNeuron; i++ {
			p.neuron[i] = &neuron{}
			//n.neuron[i].axon[0] = &axon{}
		}
		for i := 0; i < numAxon; i++ {
			p.axon[i] = &axon{
				weight:	 getRand(),
				synapse: map[string]Setter{},
			}
			//n.axon[i].synapse = make(map[string]Setter, 3)
		}*/


		/*func(index int) {
			sa, sn, pa, pn := 0, 0, 0, 0
			var cn int
			var layer HiddenType
			layer = append(p.hiddenLayer, hiddenType(lenTarget))
			for i, v := range layer {
				if i == 0 {
					cn = lenInput + b
				} else {
					cn = int(layer[i - 1]) + b
				}
				sa += cn * int(v)
				if index < sa {
					for j := 0; j < int(v); j++ { // проходим по нейронам в скрытых слоях
						delta := index - pa
						if delta < cn * (j + 1) {
							n.axon[index].synapse["output"] = n.neuron[j + sn]
							delta -= cn * j
							if i == 0 { // первый скрытый слой
								if delta < lenInput {
									if in, ok := args[0].(FloatType); ok {
										n.axon[index].synapse["input"] = floatType(in[delta])
									}
								} else { // последующие скрытые слои
									if p.bias {
										n.axon[index].synapse["bias"] = biasType(true)
									} else {
										panic("error") // !!!
									}
								}
							} else {
								n.axon[index].synapse["input"] = n.neuron[pn + delta]
							}
							fmt.Println("-", n.axon[index])
							break
						}
					}
					break
				}
				pa = sa
				pn = sn
				sn += int(v)
			}
		}(1)


		func(index int) {
			c, d, sn := 0, 0, 0
			var layer HiddenType
			layer = append(p.hiddenLayer, hiddenType(lenTarget))
			for i, v := range layer {
				for j := 0; j < int(v); j++ { // проходим по нейронам в скрытых слоях
					if i == 0 { // первый скрытый слой
						d += lenBias
					} else { // последующие скрытые слои
						d += int(p.hiddenLayer[i - 1]) + b
					}
					if d == numAxon {
						n.neuron[j + sn].axon = n.axon[c:]
					} else {
						n.neuron[j + sn].axon = n.axon[c:d]
					}
					fmt.Println("-+-", n.neuron[j + sn].axon)
					c = d
				}
				sn += int(v)
			}
		}(1)

		// Fills all weights with random numbers
		//n.setRandWeight()*/

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