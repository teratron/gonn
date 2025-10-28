package loss

import "math"

// Kullback-Leibler Divergence loss function: KLD = sum(target * log(target / predicted))
func kldLoss[T float32 | float64](predicted, target T) T {
	epsilon := T(1e-7)
	clampedPred := predicted
	if clampedPred < epsilon {
		clampedPred = epsilon
	}

	if target <= 0 {
		// If target is 0 or negative, return 0 to avoid log(0)
		return 0
	}

	ratio := target / clampedPred
	return target * T(math.Log(float64(ratio)))
}
