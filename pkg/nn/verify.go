//
package nn

import "github.com/zigenzoog/gonn/pkg"

func (n *NN) Verify(input []float64, target ...[]float64) (loss float64) {
	if !n.isTrain {
		pkg.Log("Neural network is not trained", true) // !!!
		if !n.isInit {
			if n.isInit = n.init(len(input), getLengthData(target...)...); !n.isInit {
				pkg.Log("Error initialization", true) // !!!
				return
			}
		}
	}
	if a, ok := n.Get().(NeuralNetwork); ok {
		loss = a.Verify(input, target...)
	}
	return
}