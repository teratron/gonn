package loss

import "math"

// Mean Squared Logarithmic Error loss function: MSLE = (log(predicted + 1) - log(target + 1))²
func msleLoss[T float32 | float64](predicted, target T) T {
	logPred := T(math.Log(float64(predicted + 1)))
	logTarget := T(math.Log(float64(target + 1)))
	diff := logPred - logTarget
	return diff * diff
}
