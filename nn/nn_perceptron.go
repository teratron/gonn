//
package nn

type perceptron struct {
	bias			Bias
	rate			Rate

	modeLoss		uint8		//
	levelLoss		Loss		// Minimum (sufficient) level of the average of the error during training

	numNeuronHidden	Hidden		// Array of the number of neurons in each hidden layer
	numHidden		hidden		// Number of hidden layers in the neural network

	lastIndexNeuron uint32		// Index of the output (last) layer of the neural network
	lastIndexAxon	uint32		//

	Architecture // чтобы не создавать методы для всех типов нн
}

// Initializing Perceptron Neural Network
func (n *nn) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{
		//bias:				0,
		rate:				DefaultRate,
		modeLoss:			ModeMSE,
		levelLoss:			.0001,
	}
	/*if a, ok := n.architecture.(*perceptron); ok {
		//a.numNeuronHidden = make([]uint16, 0)
		//a.numHidden = uint16(len(a.numNeuronHidden))
		//fmt.Println("*****", a.numHidden, a.numNeuronHidden)
	}*/

	return n
}

// Setter
func (p *perceptron) Set(args ...Setter) {
	switch v := args[0].(type) {
	case Bias:
		p.bias = v
	case Rate:
		p.rate = v
	case Hidden:
		p.numNeuronHidden = v
		p.numHidden = hidden(len(v))
	default:
	}
}

// Getter
func (p *perceptron) Get(args ...Getter) Getter {
	switch args[0].(type) {
	case Bias:
		return p.bias
	case Rate:
		return p.rate
	case Hidden:
		return p.numNeuronHidden
	case hidden:
		return p.numHidden
	default:
		return nil
	}
}

// Bias
func (p *perceptron) SetBias(bias Bias) {
	p.bias = bias
}

func (p *perceptron) Bias() Bias {
	return p.bias
}

func (p *perceptron) GetBias() Bias {
	return p.bias
}

// Learning rate
func (p *perceptron) SetRate(rate Rate) {
	p.rate = rate
}

func (p *perceptron) Rate() Rate {
	return p.rate
}

func (p *perceptron) GetRate() Rate {
	return p.rate
}

// Level loss

// Hidden layers
func (p *perceptron) SetHiddenLayer(args ...hidden) {
	p.numNeuronHidden = args
	p.numHidden = hidden(len(p.numNeuronHidden))
}

func (p *perceptron) GetHiddenLayer() Hidden {
	return p.numNeuronHidden
}

func (p *perceptron) GetNumHiddenLayer() hidden {
	return p.numHidden
}