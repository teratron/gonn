package loss

import (
	"github.com/teratron/gonn/pkg/utils"
	"math"
)

// Categorical Cross-Entropy loss function: CCE = -sum(target * log(predicted))
// This implementation works with vector inputs (slices)
func cceLoss[T utils.Float](predicted, target []T) T {
	if len(predicted) != len(target) {
		return 0 // Return 0 if slices have different lengths
	}

	var total T
	epsilon := T(1e-7) // Small value to prevent log(0)

	for i := 0; i < len(predicted); i++ {
		// Clamp predicted value to avoid log(0)
		clampedPred := predicted[i]
		if clampedPred < epsilon {
			clampedPred = epsilon
		} else if clampedPred > 1-epsilon {
			clampedPred = 1 - epsilon
		}

		logPred := T(math.Log(float64(clampedPred)))
		total += target[i] * logPred
	}

	return -total
}

// For single value, return 0 as CCE requires vectors
func cceLossSingle[T utils.Float](predicted, target T) T {
	return 0
}
