//
package nn

type Architecture interface {
	//
	Perceptron() NeuralNetwork

	//
	RadialBasis() NeuralNetwork

	//
	Hopfield() NeuralNetwork
}

/*func (n *NN) Hopfield() NeuralNetwork {
	n.Architecture = &hopfield.Hopfield{
		Architecture: n,
	}
	return n
}*/