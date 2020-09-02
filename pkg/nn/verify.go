//
package nn

import "github.com/zigenzoog/gonn/pkg"

func (n *nn) Verify(input []float64, target ...[]float64) (loss float64) {
	if !n.IsTrain {
		pkg.Log("Neural network is not trained", true) // !!!
		if !n.IsInit {
			if n.IsInit = n.init(len(input), getLengthData(target...)...); !n.IsInit {
				pkg.Log("Error initialization", true) // !!!
				return
			}
		}
	}
	if a, ok := n.Architecture.(NeuralNetwork); ok {
		loss = a.Verify(input, target...)
	}
	return
}
