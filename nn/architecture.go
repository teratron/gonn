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
func getArchitecture(net Architecture) (NeuralNetwork, bool) {
	if n, ok := net.(*nn); ok {
		if v, ok := n.Architecture.(NeuralNetwork); ok {
			return v, ok
		}
	}
	return nil, false
}

//func returnP()