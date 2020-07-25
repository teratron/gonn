package nn

import "io"

func (n *nn) Write(writer io.Writer) {
	if a, ok := n.Get().(NeuralNetwork); ok {
		a.Write(writer)
	}
}