package nn

//
func (n *nn) Query(input []float64) (output []float64) {
	if !n.isTrain {
		Log("Neural network is not trained", true) // !!!
		if !n.isInit {
			Log("Error initialization", true) // !!!
			return nil
		}
	}
	if a, ok := n.Get().(NeuralNetwork); ok {
		output = a.Query(input)
	}
	return
}