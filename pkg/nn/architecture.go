//
package nn

type Architecture interface {
	//
	//Perceptron() NeuralNetwork
	Perceptron

	//
	//RadialBasis() NeuralNetwork
	RadialBasis

	//
	//Hopfield() NeuralNetwork
	Hopfield
}

//
/*func getArchitecture(net Architecture) (NeuralNetwork, bool) {
	if n, ok := net.(*nn).Architecture.(NeuralNetwork); ok {
		return n, ok
	}
	return nil, false
}*/