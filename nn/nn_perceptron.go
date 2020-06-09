//
package nn

type Perceptron struct {
	Specifier
}

func (n NeuralNetwork) Perceptron() NeuralNetwork {
	n.Specifier = &Perceptron{}
	return n
}

func (p *Perceptron) Set(setter Setter) {
}

func (p *Perceptron) Get() Getter {
	return p
}