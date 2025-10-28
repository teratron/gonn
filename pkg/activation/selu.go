package activation

import "math"

// SeLU activation function: f(x) = x < 0 ? scale * (alpha * (exp(x) - 1)) : scale * x
func seluActivation[T float32 | float64](value T, scale, alpha float64) T {
	if value < T(0) {
		return T(scale) * (T(alpha) * (T(math.Exp(float64(value))) - T(1.0)))
	} else {
		return T(scale) * value
	}
}

// SeLU derivative function
func seluDerivative[T float32 | float64](value T, scale, alpha float64) T {
	if value < T(0) {
		// Derivative for x < 0 is scale * alpha * exp(x)
		return T(scale) * T(alpha) * T(math.Exp(float64(value)))
	} else {
		// Derivative for x >= 0 is scale
		return T(scale)
	}
}
