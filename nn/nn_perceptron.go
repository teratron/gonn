//
package nn

type perceptron struct {
	NeuralNetwork
}

func (n *neuralNetwork) Perceptron() NeuralNetwork {
	n.NeuralNetwork = &perceptron{}
	return n
}

func (p *perceptron) Set(setter Setter) {
}

func (p *perceptron) Get() Getter {
	return p
}