//
package nn

type feedForward struct {
	bias		Bias
	hiddenLayer	[]uint32
	Architecture // чтобы не создавать методы для всех типов нн
	Parameter
}

// Initializing Feed Forward Neural Network
func (n *NN) FeedForward() NeuralNetwork {
	n.architecture = &feedForward{
		bias: .1,
	}
	//n.network.(*feedForward).Bias = 1
	return n
}

func (f *feedForward) Set(args ...Setter) {
	switch v := args[0].(type) {
	case Bias:
		f.bias = v
	/*case Rate:
		f.rate = v*/
	default:
	}
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