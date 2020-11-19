package nn

import "fmt"

// Verify
func (n *nn) Verify(input []float64, target ...[]float64) (loss float64) {
	if !n.isTrain {
		errNN(fmt.Errorf("verify: %w", ErrNotTrained))
		if !n.isInit {
			if n.isInit = n.init(len(input), getLengthData(target...)...); !n.isInit {
				errNN(fmt.Errorf("%w for verify", ErrInit))
				return
			}
		}
	}
	if a, ok := n.Architecture.(NeuralNetwork); ok {
		loss = a.Verify(input, target...)
	}
	return
}
