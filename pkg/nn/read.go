package nn

import "io"

func (n *nn) Read(reader io.Reader) {
	if a, ok := n.Get().(NeuralNetwork); ok {
		a.Read(reader)
	}
}