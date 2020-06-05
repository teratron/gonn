//
package nn

type Perceptron struct {
	Architecture
}

func (n NeuralNetwork) Perceptron() NeuralNetwork {
	n.Architecture = Perceptron{}
	return n
}
