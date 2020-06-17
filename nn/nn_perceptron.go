//
package nn

type perceptron struct {
	bias			Bias
	rate			Rate
	modeLoss		uint8
	limitLoss		Loss		// Minimum (sufficient) level of the average of the error during training
	numHidden		uint16		// Number of hidden layers in the neural network
	numNeuronHidden	Hidden		// Array of the number of neurons in each hidden layer
	lastIndexNeuron uint32		// Index of the output (last) layer of the neural network
	lastIndexAxon	uint32		//

	Architecture // чтобы не создавать методы для всех типов нн
}

// Initializing Perceptron Neural Network
func (n *NN) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{
		//bias:				0,
		rate:				DefaultRate,
		modeLoss:			ModeMSE,
		limitLoss:			.0001,
		//numHidden:			0,
		//numNeuronHidden:	nil, //[]uint16{},
	}
	/*if a, ok := n.architecture.(*perceptron); ok {
		//a.numNeuronHidden = make([]uint16, 0)
		//a.numHidden = uint16(len(a.numNeuronHidden))
		//fmt.Println("*****", a.numHidden, a.numNeuronHidden)
	}*/

	return n
}

func (p *perceptron) Set(args ...Setter) {
	switch v := args[0].(type) {
	case Bias:
		p.bias = v
	case Rate:
		p.rate = v
	case Hidden:
		p.numNeuronHidden = v
	default:
	}
}

func (p *perceptron) Get(args ...Getter) Getter {
	panic("implement me")
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

// Limit loss

// Number of neurons in each hidden layer
func (p *perceptron) SetHiddenLayer(args ...hidden) {
	p.numNeuronHidden = args
}

func (p *perceptron) GetHiddenLayer() Hidden {
	return p.numNeuronHidden
}