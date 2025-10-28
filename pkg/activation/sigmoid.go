package activation

import "math"

// Sigmoid activation function: f(x) = slope / (exp(-x) + 1)
func sigmoidActivation[T float32 | float64](value T, slope float64) T {
	return T(slope) / (T(math.Exp(float64(-value))) + T(1.0))
}

// Sigmoid derivative function
func sigmoidDerivative[T float32 | float64](value T, slope float64) T {
	sigmoidVal := sigmoidActivation(value, slope)
	return sigmoidVal * (T(1.0) - sigmoidVal/T(slope))
}
