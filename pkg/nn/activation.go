package nn

import "github.com/teratron/gonn/pkg/params"

// Activation function.
func Activation(value float64, mode uint8) float64 {
	return params.Activation(value, mode)
}

// Derivative activation function.
func Derivative(value float64, mode uint8) float64 {
	return params.Derivative(value, mode)
}
