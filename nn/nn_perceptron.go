// Perceptron Neural Network
package nn

type Perceptron interface {
	Perceptron() NeuralNetwork
}

type perceptron struct {
	bias biasType
	rate rateType

	modeLoss  uint8    //
	levelLoss lossType // Minimum (sufficient) level of the average of the error during training

	hiddenLayer HiddenType // Array of the number of neurons in each hidden layer

	lastIndexNeuron uint32		// Index of the output (last) layer of the neural network
	lastIndexAxon	uint32		//

	Architecture // чтобы не создавать методы для всех типов нн
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
func (p *perceptron) Set(set ...Setter) {
	switch v := set[0].(type) {
	case biasType:
		p.bias = v
	case rateType:
		p.rate = v
	case lossType:
		p.levelLoss = v
	case HiddenType:
		p.hiddenLayer = v
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
	case lossType:
		return p.levelLoss
	case HiddenType:
		return p.hiddenLayer
	default:
		Log("This type of variable is missing for Perceptron Neural Network", false)
		return nil
	}
}