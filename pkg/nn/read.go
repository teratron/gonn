package nn

import "io"

func (n *NN) Read(reader io.Reader) {
	if a, ok := n.Get().(NeuralNetwork); ok {
		a.Read(reader)
	}
}