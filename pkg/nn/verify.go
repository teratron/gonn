//
package nn

func (n *NN) Verify(input []float64, target ...[]float64) (loss float64) {
	if !n.IsTrain {
		Log("Neural network is not trained", true) // !!!
		if !n.IsInit {
			if n.IsInit = n.init(input, target...); !n.IsInit {
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