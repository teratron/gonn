//
package nn

type feedForward struct {
	bias		Bias
	hiddenLayer	[]uint32
	Architecture // чтобы не создавать методы для всех типов нн
}

// Initializing Feed Forward Neural Network
func (n *NN) FeedForward() NeuralNetwork {
	n.architecture = &feedForward{
		bias: .1,
	}
	//n.network.(*feedForward).Bias = 1
	return n
}

func (f *feedForward) Set(arg GetterSetter) {
	switch v := arg.(type) {
	case Bias:
		f.bias = v
	default:
	}
}

func (f *feedForward) Get() GetterSetter {
	return f
}

// Bias
func (f *feedForward) SetBias(bias Bias) {
	f.Set(bias)
}

func (f *feedForward) Bias() Bias {
	return f.bias
}

func (f *feedForward) GetBias() Bias {
	return f.Bias()
}

// Rate
func (f *feedForward) Rate() Rate {
	panic("implement me")
}

func (f *feedForward) GetRate() Rate {
	panic("implement me")
}

func (f *feedForward) SetRate(Rate) {
	panic("implement me")
}