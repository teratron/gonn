package loss

import "math"

// Poisson Loss Function: Poisson = predicted - target * log(predicted)
func poissonLoss[T float32 | float64](predicted, target T) T {
	epsilon := T(1e-7)
	clampedPred := predicted
	if clampedPred < epsilon {
		clampedPred = epsilon
	}

	logPred := T(math.Log(float64(clampedPred)))
	return clampedPred - target*logPred
}
