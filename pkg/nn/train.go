//
package nn

const MaxIteration uint = 10e+05	// The maximum number of iterations after which training is forcibly terminated

func (n *nn) Train(data ...[]float64) (loss float64, count int) {
	if len(data) > 0 {
		if !n.isInit {
			if n.isInit = n.init(data...); !n.isInit {
				Log("Error initialization", true) // !!!
				return
			}
		}
		if a, ok := n.Get().(NeuralNetwork); ok {
			loss, count = a.Train(data...)
			if count > 0 {
				n.isQuery = true
				n.isTrain = true
			}
		}
	} else {
		Log("Empty Train()", true) // !!!
	}
	return
}