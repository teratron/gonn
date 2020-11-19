package nn

import "fmt"

// MaxIteration the maximum number of iterations after which training is forcibly terminated
const MaxIteration int = 10e+05

// Train
func (n *nn) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if !n.isInit {
		if n.isInit = n.init(len(input), getLengthData(target...)...); !n.isInit {
			errNN(fmt.Errorf("%w for train", ErrInit))
			return
		}
	}
	if a, ok := n.Architecture.(NeuralNetwork); ok {
		loss, count = a.Train(input, target...)
		if count > 0 {
			n.isTrain = true
		}
	}
	return
}
