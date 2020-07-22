//
package nn

func (n *nn) Verify(input []float64, target ...[]float64) (loss float64) {
	if !n.isTrain {
		Log("Neural network is not trained", true) // !!!
		if !n.isInit {
			if n.isInit = n.init(input, target...); !n.isInit {
				Log("Error initialization", true) // !!!
				return
			}
		}
	}
	if a, ok := n.Get().(NeuralNetwork); ok {
		loss = a.Verify(input, target...)
	}
	return
}