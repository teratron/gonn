//
package nn

const MaxIteration uint32 = 10e+05	// The maximum number of iterations after which training is forcibly terminated

func (n *nn) Train(data ...[]float64) (loss float64, count int) {
	if !n.isInit {
		if n.isInit = n.init(floatArrayType(data[0]), floatArrayType(data[1])); !n.isInit {
			Log("Error initialization", true) // !!!
			return
		}
	}
	if a, ok := n.Get().(NeuralNetwork); ok {
		loss, count = a.Train(data[0], data[1])
		if count > 0 {
			n.isQuery = true
			n.isTrain = true
		}
	}
	return
}