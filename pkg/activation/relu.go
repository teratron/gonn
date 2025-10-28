package activation

// ReLU activation function: f(x) = x < 0 ? leak * x : x
func reluActivation[T float32 | float64](value T, leak float64) T {
	if value < T(0) {
		return value * T(leak)
	} else {
		return value
	}
}

// ReLU derivative function
func reluDerivative[T float32 | float64](value T, leak float64) T {
	if value < T(0) {
		return T(leak)
	} else {
		return T(1.0)
	}
}
