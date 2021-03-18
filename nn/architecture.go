package nn

import zoo "github.com/teratron/gonn/nn/architecture"

func architecture(name ...string) NeuralNetwork {
	return zoo.Get(name[0])
}

// Perceptron
func Perceptron() NeuralNetwork {
	return zoo.Get(zoo.Perceptron)
}

// Hopfield
func Hopfield() NeuralNetwork {
	return zoo.Get(zoo.Hopfield)
}
