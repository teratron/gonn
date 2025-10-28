package activation

import "math"

// Softmax activation function (simplified for single value)
func softmaxActivation[T float32 | float64](value T) T {
	// This is a placeholder. The actual implementation would require a vector of values.
	expVal := T(math.Exp(float64(value)))
	// In a real scenario, we would need the sum of exps for all values in the layer.
	// For a single value, this is not a meaningful softmax.
	// Let's assume a simplification where it normalizes against a hypothetical total of 1.
	return expVal / (expVal + T(1.0)) // This is essentially a type of sigmoid
}

// Softmax derivative function (simplified for single value)
func softmaxDerivative[T float32 | float64](value T) T {
	// The derivative of softmax is more complex and depends on the output vector.
	// For a single value, we can approximate with the derivative of the simplified activation.
	activationVal := softmaxActivation(value)
	return activationVal * (T(1.0) - activationVal)
}
