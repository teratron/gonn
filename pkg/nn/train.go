//
package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

// The maximum number of iterations after which training is forcibly terminated
const MaxIteration uint = 10e+05

// Train
func (n *nn) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if !n.isInit {
		if n.isInit = n.init(len(input), getLengthData(target...)...); !n.isInit {
			pkg.Log("Error initialization", true) // !!!
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