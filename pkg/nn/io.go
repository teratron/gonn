package nn

import "io"

// Output of neural network training results in io.Writer
func (n *nn) Print(writer io.Writer, input []float64, args ...interface{}) {
	if a, ok := n.Get().(NeuralNetwork); ok {
		a.Print(writer, input, args...)
	}
}
