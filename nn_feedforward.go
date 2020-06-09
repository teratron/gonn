//
package main

type FeedForward struct {
	Bias

	HiddenLayer []uint32

	Specifier // чтобы не создавать методы для всех типов нн
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
	switch v := setter.(type) {
	case Bias:
		f.Bias = v
	default:
	}
}

func (f *FeedForward) Get() Getter {
	return f
}
