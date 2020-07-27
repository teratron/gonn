//
package nn

func (n *NN) Query(input []float64) (output []float64) {
	if !n.IsTrain {
		Log("Neural network is not trained", true) // !!!
		if !n.IsInit {
			Log("Error initialization", true) // !!!
			return nil
		}
	}
	if a, ok := n.Get().(NeuralNetwork); ok {
		output = a.Query(input)
	}
	return
}