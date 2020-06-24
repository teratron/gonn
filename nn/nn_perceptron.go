// Perceptron Neural Network
package nn

type Perceptron interface {
	Perceptron() NeuralNetwork
}

type perceptron struct {
	bias			biasType			//
	rate			rateType			//
	modeActivation	modeActivationType	//

	modeLoss		modeLossType		//
	levelLoss		lossType			// Minimum (sufficient) level of the average of the error during training

	hiddenLayer		HiddenType			// Array of the number of neurons in each hidden layer

	upperRange		floatType			// Range, Bound, Limit, Scope
	lowerRange		floatType

	lastNeuron		uint32				// Index of the last neuron of the neural network
	lastAxon		uint32				// Index of the last axon of the neural network

	neuron struct {
		error		floatType
	}

	Architecture // чтобы не создавать методы для всех типов нн
}

// Initializing Perceptron Neural Network
func (n *nn) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{
		bias:			false,
		rate:			DefaultRate,
		modeActivation:	ModeSIGMOID,
		modeLoss:		ModeMSE,
		levelLoss:		.0001,
		hiddenLayer:	HiddenType{},
		upperRange:		1,
		lowerRange:		0,
		lastNeuron:		0,
		lastAxon:		0,
	}
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
	case lossType:
		p.levelLoss = v
	case HiddenType:
		p.hiddenLayer = v
		//fmt.Println(*p)
		//p.initHidden()
		/*fmt.Println("***", func(d uint32) (h hiddenType) {
			h = 1
			for _, value := range p.hiddenLayer {
				h *= value
			}
			return h + hiddenType(d)
		}(4))*/
	default:
		Log("This type of variable is missing for Perceptron Neural Network", false)
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
	case lossType:
		return p.levelLoss
	case HiddenType:
		return p.hiddenLayer
	default:
		Log("This type of variable is missing for Perceptron Neural Network", false)
		return nil
	}
}

func (p *perceptron) initHidden() {
}