package loss

import "math"

// Root Mean Squared Error loss function: RMSE = sqrt((predicted - target)²)
// For single values, this is equivalent to absolute error, but in vector context,
// it's the square root of the mean of squared errors
func rmseLoss[T float32 | float64](predicted, target T) T {
	diff := predicted - target
	return T(math.Sqrt(float64(diff * diff)))
}
