package params

import "math"

// Activation function mode.
const (
	LINEAR    uint8 = iota // Linear/identity.
	RELU                   // ReLu (rectified linear unit).
	LEAKYRELU              // Leaky ReLu (leaky rectified linear unit).
	SIGMOID                // Logistic, a.k.a. sigmoid or soft step.
	TANH                   // TanH (hyperbolic tangent).
)

// CheckActivationMode.
func CheckActivationMode(mode uint8) uint8 {
	if mode > TANH {
		return SIGMOID
	}
	return mode
}

// Activation function.
func Activation(value float64, mode uint8) float64 {
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
		return 1 / (1 + math.Exp(-value))
	case TANH:
		value = math.Exp(2 * value)
		return (value - 1) / (value + 1)
	}
}

// Derivative activation function.
func Derivative(value float64, mode uint8) float64 {
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
		return 1 - math.Pow(value, 2)
	}
}
