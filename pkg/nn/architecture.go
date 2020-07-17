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