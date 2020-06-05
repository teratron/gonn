//
package nn

type FeedForward struct {
	Bias

	HiddenLayer []float32

	Architecture
}

func (n NeuralNetwork) FeedForward() NeuralNetwork {
	n.Architecture = FeedForward{
		Bias: 1,
		//HiddenLayer:
	}
	return n
}

/*func (f *FeedForward) Set(s Setter) {

}*/
