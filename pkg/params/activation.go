package params

import (
	"math"

	"github.com/teratron/gonn/pkg"
)

// Activation function mode.
const (
	LINEAR    uint8 = iota // LINEAR - Linear/identity.
	RELU                   // RELU - ReLu (rectified linear unit).
	LEAKYRELU              // LEAKYRELU - Leaky ReLu (leaky rectified linear unit).
	SIGMOID                // SIGMOID - Logistic, a.k.a. sigmoid or soft step.
	TANH                   // TANH - TanH (hyperbolic tangent).
)

// CheckActivationMode.
func CheckActivationMode(mode uint8) uint8 {
	if mode > TANH {
		return SIGMOID
	}
	return mode
}

// Activation function.
func Activation(value pkg.FloatType, mode uint8) pkg.FloatType {
	switch mode {
	case LINEAR:
		return value
	case RELU:
		if value < 0 {
			return 0
		}
		return value
	case LEAKYRELU:
		if value < 0 {
			return .01 * value
		}
		return value
	default:
		fallthrough
	case SIGMOID:
		return 1 / (1 + pkg.FloatType(math.Exp(float64(-value))))
	case TANH:
		value = pkg.FloatType(math.Exp(2 * float64(value)))
		return (value - 1) / (value + 1)
	}
}

// Derivative activation function.
func Derivative(value pkg.FloatType, mode uint8) pkg.FloatType {
	switch mode {
	case LINEAR:
		return 1
	case RELU:
		if value < 0 {
			return 0
		}
		return 1
	case LEAKYRELU:
		if value < 0 {
			return .01
		}
		return 1
	default:
		fallthrough
	case SIGMOID:
		return value * (1 - value)
	case TANH:
		return 1 - pkg.FloatType(math.Pow(float64(value), 2))
	}
}
