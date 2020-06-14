//
package nn

type perceptron struct {
	Architecture
	Parameter
}

func (n *NN) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{}
	return n
}

func (p *perceptron) Set(args ...Setter) {
}
