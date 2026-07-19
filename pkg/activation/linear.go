package activation

import "github.com/teratron/gonn/pkg/utils"

// Linear activation function: f(x) = slope * x + offset
func linearActivation[T utils.Float](value T, slope, offset float64) T {
	return value*T(slope) + T(offset)
}

// Linear derivative function
func linearDerivative[T utils.Float](slope float64) T {
	return T(slope)
}
