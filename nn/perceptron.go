package nn

type Perceptron struct {
	Typer
}

func (n NeuralNetwork) Perceptron() NeuralNetwork {
	n.Architecture = Perceptron{}
	return n
}