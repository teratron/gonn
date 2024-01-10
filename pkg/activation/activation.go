package activation

import (
	"log"

	. "github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

type Type uint8

// Activation function mode.
const (
	LINEAR    Type = iota // LINEAR - Linear/identity.
	RELU                  // RELU - ReLu (rectified linear unit).
	LEAKYRELU             // LEAKYRELU - Leaky ReLu (leaky rectified linear unit).
	SIGMOID               // SIGMOID - Logistic, a.k.a. sigmoid or soft step.
	TANH                  // TANH - TanH (hyperbolic tangent).
)

// CheckActivationMode.
func CheckActivationMode(mode Type) Type {
	if mode > TANH {
		return SIGMOID
	}

	return mode
}

// Activation function.
func Activation[T Floater](value T, mode Type) T {
	switch mode {
	case LINEAR:
		return value
	case RELU:
		if value < 0. {
			return 0.
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
		value = 1 / (1 + utils.Exp(-value))
		switch { // TODO:
		case utils.IsNaN(value):
			log.Panic("SIGMOID: loss not-a-number value") // TODO: log.Panic (?)
		case utils.IsInf(value, 0):
			log.Panic("SIGMOID: loss is infinity") // TODO: log.Panic (?)
		}
		return value
		//return 1 / (1 + Exp(-value))
	case TANH:
		val0 := value
		value = utils.Exp(2 * value)
		val1 := value
		value = (value - 1) / (value + 1)
		switch { // TODO:
		case utils.IsNaN(value):
			log.Panic("TANH: loss not-a-number value", val0, val1) // TODO: log.Panic (?)
		case utils.IsInf(value, 0):
			log.Panic("TANH: loss is infinity") // TODO: log.Panic (?)
		}
		return value
		//value = Exp(2 * value)
		//return (value - 1) / (value + 1)
	}
}

// Derivative activation function.
func Derivative[T Floater](value T, mode Type) T {
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
		return 1 - utils.Pow(value, 2)
	}
}
