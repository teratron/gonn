package loss

import "math"

// Log-Cosh Loss: LOG_COSH = log(cosh(predicted - target))
func logCoshLoss[T float32 | float64](predicted, target T) T {
	diff := predicted - target
	return T(math.Log(math.Cosh(float64(diff))))
}
