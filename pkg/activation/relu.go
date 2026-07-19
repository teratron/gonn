package activation

import "github.com/teratron/gonn/pkg/utils"

// ReLU activation function: f(x) = x < 0 ? leak * x : x
func reluActivation[T utils.Float](value T, leak float64) T {
	if value < T(0) {
		return value * T(leak)
	} else {
		return value
	}
}

// ReLU derivative function
func reluDerivative[T utils.Float](value T, leak float64) T {
	if value < T(0) {
		return T(leak)
	} else {
		return T(1.0)
	}
}
