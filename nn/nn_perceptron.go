//
package nn

type perceptron struct {
	Architecture
	Parameter
}

func (n *zzNN) Perceptron() NeuralNetwork {
	n.architecture = &perceptron{}
	return n
}

func (p *perceptron) Set(setter Setter) {
}

func (p *perceptron) Get() Getter {
	return p
}
