package nn

import (
	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
)

// Activation function.
func Activation(value float64, mode uint8) float64 {
	return float64(params.Activation(pkg.FloatType(value), mode))
}

// Derivative activation function.
func Derivative(value float64, mode uint8) float64 {
	return float64(params.Derivative(pkg.FloatType(value), mode))
}
