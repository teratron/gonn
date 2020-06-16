//
package nn

type perceptron struct {
	bias		Bias
	rate		Rate
	hiddenLayer	[]uint32
	Architecture // чтобы не создавать методы для всех типов нн
}

// Initializing Feed Forward Neural Network
func (n *NN) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{
		bias: 1,
		rate: DefaultRate,
	}
	//n.network.(*perceptron).Bias = 1
	return n
}

func (p *perceptron) Set(args ...Setter) {
	switch v := args[0].(type) {
	case Bias:
		p.bias = v
	case Rate:
		p.rate = v
	default:
	}
}

func (p *perceptron) Get(args ...Getter) Getter {
	panic("implement me")
}

// Bias
func (p *perceptron) SetBias(bias Bias) {
	p.Set(bias)
}

func (p *perceptron) Bias() Bias {
	return p.bias
}

func (p *perceptron) GetBias() Bias {
	return p.bias
}

// Learning rate
func (p *perceptron) SetRate(rate Rate) {
	p.Set(rate)
}

func (p *perceptron) Rate() Rate {
	return p.rate
}

func (p *perceptron) GetRate() Rate {
	return p.rate
}