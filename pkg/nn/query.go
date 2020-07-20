package nn

//
func (n *nn) Query(input []float64) (output []float64) {
	if !n.isInit {
		Log("An uninitialized neural network", true)
		return nil
	}
	if a, ok := n.Get().(NeuralNetwork); ok {
		output = a.Query(input)
		/*if count > 0 {
			n.isQuery = true
			n.isTrain = true
		}*/
	}
	return
}