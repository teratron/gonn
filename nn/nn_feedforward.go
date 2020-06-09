//
package nn

type feedForward struct {
	bias

	HiddenLayer []uint32

	NeuralNetwork // чтобы не создавать методы для всех типов нн
}

func (n *neuralNetwork) FeedForward() NeuralNetwork {
	n.NeuralNetwork = &feedForward{
		bias: 1,
		//HiddenLayer:
	}
	//n.Architecture.(*feedForward).Bias = 1
	return n
}

func (f *feedForward) Set(setter Setter) {
	switch v := setter.(type) {
	case bias:
		f.bias = v
	default:
	}
}

func (f *feedForward) Get() Getter {
	return f
}
