package loss

import "math"

// Arctan Error loss function: ARCTAN = arctan(predicted - target)
func arctanLoss[T float32 | float64](predicted, target T) T {
	return T(math.Atan(float64(predicted - target)))
}
