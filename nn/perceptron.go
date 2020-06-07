//
package nn

type Perceptron struct {
	Specifier
}

func (n NeuralNetwork) Perceptron() NeuralNetwork {
	n.Specifier = &Perceptron{}
	return n
}

func (f *Perceptron) Set(setter Setter) {
}
