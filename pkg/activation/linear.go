package activation

// Linear activation function: f(x) = slope * x + offset
func linearActivation[T float32 | float64](value T, slope, offset float64) T {
	return value*T(slope) + T(offset)
}

// Linear derivative function
func linearDerivative(slope float64) float64 {
	return slope
}
