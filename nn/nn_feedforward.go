//
package nn

type feedForward struct {
	bias		Bias
	hiddenLayer	[]uint32
	Architecture // чтобы не создавать методы для всех типов нн
}

func (n *zzNN) FeedForward() NeuralNetwork {
	n.architecture = &feedForward{
		bias: .1,
	}
	//n.network.(*feedForward).Bias = 1
	return n
}

func (f *feedForward) Set(setter Setter) {
	switch v := setter.(type) {
	case Bias:
		f.bias = v
	default:
	}
}

func (f *feedForward) Get() Getter {
	return f
}

func (f *feedForward) Bias() Bias {
	return f.bias
}

func (f *feedForward) GetBias() Bias {
	return f.Bias()
}
