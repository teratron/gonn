package nn

import "fmt"

// Verify
func (n *nn) Verify(input []float64, target ...[]float64) (loss float64) {
	if !n.IsTrain {
		errNN(fmt.Errorf("verify: %w", ErrNotTrained))
		if !n.IsInit {
			if n.IsInit = n.init(len(input), getLengthData(target...)...); !n.IsInit {
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