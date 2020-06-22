// Perceptron Neural Network
package nn

type Perceptron interface {
	Perceptron() NeuralNetwork
	//BiasInt

	//
	GetRate() RateType
	SetRate(RateType)

	//
	GetHidden() Hidden
	SetHidden(...hidden)
}

type perceptron struct {
	bias Bias
	rate RateType

	modeLoss		uint8		//
	levelLoss		Loss		// Minimum (sufficient) level of the average of the error during training

	hiddenLayer		Hidden		// Array of the number of neurons in each hidden layer

	lastIndexNeuron uint32		// Index of the output (last) layer of the neural network
	lastIndexAxon	uint32		//

	Architecture // чтобы не создавать методы для всех типов нн
	//Parameter
	//Perceptron
}

// Initializing Perceptron Neural Network
func (n *nn) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{
		rate:		DefaultRate,
		modeLoss:	ModeMSE,
		levelLoss:	.0001,
	}

	return n
}

// Setter
func (p *perceptron) Set(args ...Setter) {
	switch v := args[0].(type) {
	case Bias:
		p.bias = v
	case RateType:
		p.rate = v
	case Hidden:
		p.hiddenLayer = v
	default:
	}
}

// Getter
func (p *perceptron) Get(args ...Getter) Getter {
	switch args[0].(type) {
	case Bias:
		return p.bias
	case RateType:
		return p.rate
	case Hidden:
		return p.hiddenLayer
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
func (p *perceptron) SetRate(rate RateType) {
	p.rate = rate
}

func (p *perceptron) GetRate() RateType {
	return p.rate
}

// Level loss

// Hidden layers
func (p *perceptron) SetHidden(args ...hidden) {
	p.hiddenLayer = args
}

func (p *perceptron) GetHidden() Hidden {
	return p.hiddenLayer
}