package nn

import arch "github.com/teratron/gonn/architecture"

func architecture(name ...string) NeuralNetwork {
	/*return &NN{
		NeuralNetwork: arch.Architecture(name[0]),
	}*/
	return arch.Architecture(name[0])
}

// Perceptron
func Perceptron() NeuralNetwork {
	/*return &NN{
		NeuralNetwork: arch.Architecture(arch.Perceptron),
	}*/
	return arch.Architecture(arch.Perceptron)
}

// Hopfield
func Hopfield() NeuralNetwork {
	/*return &NN{
		NeuralNetwork: arch.Architecture(arch.Hopfield),
	}*/
	return arch.Architecture(arch.Hopfield)
}
