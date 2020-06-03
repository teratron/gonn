//
package nn

type FeedForward struct {
	Bias
	Typer
}

func (n NeuralNetwork) FeedForward() NeuralNetwork {
	n.Architecture = FeedForward{
		Bias: 1,
	}
	return n
}
