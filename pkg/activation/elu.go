package activation

import "math"

// ELU activation function: f(x) = x < 0 ? alpha * (exp(x) - 1) : x
func eluActivation[T float32 | float64](value T, alpha float64) T {
	if value < T(0) {
		return T(alpha) * (T(math.Exp(float64(value))) - T(1.0))
	} else {
		return value
	}
}

// ELU derivative function
func eluDerivative[T float32 | float64](value T, alpha float64) T {
	if value < T(0) {
		// Derivative for x < 0 is alpha * exp(x)
		return T(alpha) * T(math.Exp(float64(value)))
	} else {
		// Derivative for x >= 0 is 1
		return T(1.0)
	}
}
