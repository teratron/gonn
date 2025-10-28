package loss

import "math"

// Binary Cross-Entropy loss function: BCE = -(target * log(predicted) + (1 - target) * log(1 - predicted))
func bceLoss[T float32 | float64](predicted, target T) T {
	// Clamp predicted to avoid log(0) which would result in -inf
	epsilon := T(1e-7)
	clampedPred := predicted
	if clampedPred < epsilon {
		clampedPred = epsilon
	} else if clampedPred > 1-epsilon {
		clampedPred = 1 - epsilon
	}

	return -(target*T(math.Log(float64(clampedPred))) + (1-target)*T(math.Log(float64(1-clampedPred))))
}
