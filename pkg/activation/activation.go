package activation

import (
	"github.com/teratron/gonn/pkg/nn"
	"log"
	"math"
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
func CalcActivation[T float32 | float64](value T, mode Type) T {
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
		value = 1 / (1 + math.Exp(-value))
		switch { // TODO:
		case math.IsNaN(value):
			log.Panic("SIGMOID:perceptron.NN.calcLoss: loss not-a-number value") // TODO: log.Panic (?)
		case math.IsInf(value, 0):
			log.Panic("SIGMOID:perceptron.NN.calcLoss: loss is infinity") // TODO: log.Panic (?)
		}
		return value
		//return 1 / (1 + pkg.FloatType(math.Exp(float64(-value))))
	case TANH:
		val0 := value
		value = nn.FloatType(math.Exp(2 * float64(value)))
		val1 := value
		value = (value - 1) / (value + 1)
		switch { // TODO:
		case IsNaN(float64(value)):
			log.Panic("TANH:perceptron.NN.calcLoss: loss not-a-number value", val0, val1) // TODO: log.Panic (?)
		case math.IsInf(float64(value), 0):
			log.Panic("TANH:perceptron.NN.calcLoss: loss is infinity") // TODO: log.Panic (?)
		}
		return value
		//value = pkg.FloatType(math.Exp(2 * float64(value)))
		//return (value - 1) / (value + 1)
	}
}

// Derivative activation function.
func CalcDerivative(value nn.FloatType, mode Type) nn.FloatType {
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
		return 1 - nn.FloatType(math.Pow(float64(value), 2))
	}
}

func IsNaN[T nn.Floater](value T) bool {
	//switch f := nn.Floater(value).(type) {
	//case float32:
	//	return math.IsNaN(float64(f))
	//case float64:
	//	return math.IsNaN(f)
	//default:
	//	return false
	//}

	if v, ok := any(value).(float32); ok {
		return math.IsNaN(float64(v))
	}

	return math.IsNaN(float64(value))
}
