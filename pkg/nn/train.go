//
package nn

// The maximum number of iterations after which training is forcibly terminated
const MaxIteration uint = 10e+05

func (n *NN) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if !n.IsInit {
		if n.IsInit = n.init(input, target...); !n.IsInit {
			Log("Error initialization", true) // !!!
			return
		}
	}
	if a, ok := n.Get().(NeuralNetwork); ok {
		loss, count = a.Train(input, target...)
		if count > 0 {
			n.IsTrain = true
		}
	}
	return
}