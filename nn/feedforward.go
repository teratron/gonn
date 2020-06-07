//
package nn

type FeedForward struct {
	Bias

	HiddenLayer []uint16

	Specifier
}

func (n NeuralNetwork) FeedForward() NeuralNetwork {
	n.Specifier = &FeedForward{
		Bias: 1,
		//HiddenLayer:
	}
	//n.Specifier.(*FeedForward).Bias = 1
	return n
}

func (f *FeedForward) Set(setter Setter) {
	f.Bias = setter.(Bias)
}
