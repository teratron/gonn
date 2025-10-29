package activation

import (
	"github.com/teratron/gonn/pkg/utils"
)

// ActivationType represents different activation functions.
type ActivationType uint8

// Activation function mode.
const (
	ELISH     ActivationType = iota // ELISH - Exponential Linear Unit + Sigmoid.
	ELU                             // ELU - Exponential Linear Unit.
	LINEAR                          // LINEAR - Linear/identity.
	LEAKYRELU                       // LEAKYRELU - Leaky ReLu (leaky rectified linear unit).
	RELU                            // RELU - ReLu (rectified linear unit).
	SELU                            // SELU - Scaled Exponential Linear Unit.
	SIGMOID                         // SIGMOID - Logistic, a.k.a. sigmoid or soft step.
	SOFTMAX                         // SOFTMAX - Softmax (Note: Current implementation is a placeholder for single values, requires vector for full functionality).
	SWISH                           // SWISH - Swish-function.
	TANH                            // TANH - TanH (hyperbolic tangent).
)

type ActivationFunction[T utils.Float] interface {
	activation(value T) T
	derivative(value T) T
}

// Activation function with parameters.
func Activation[T utils.Float](value T, mode ActivationType, params ...float64) T {
	switch mode {
	case ELISH:
		return T(elishActivation(float64(value)))
	case ELU:
		alpha := 1.0
		if len(params) > 0 {
			alpha = params[0]
		}
		return T(eluActivation(value, alpha))
	case LINEAR:
		slope := 1.0
		offset := 0.0
		if len(params) > 0 {
			slope = params[0]
		}
		if len(params) > 1 {
			offset = params[1]
		}
		return T(linearActivation(value, slope, offset))
	case LEAKYRELU:
		leak := 0.01
		if len(params) > 0 {
			leak = params[0]
		}
		return T(reluActivation(value, leak))
	case RELU:
		return T(reluActivation(value, 0.0))
	case SELU:
		scale := 1.0507009873554804934193349852946
		alpha := 1.6732632423543772848170429916717
		if len(params) > 0 {
			scale = params[0]
		}
		if len(params) > 1 {
			alpha = params[1]
		}
		return T(seluActivation(value, scale, alpha))
	case SIGMOID:
		slope := 1.0
		if len(params) > 0 {
			slope = params[0]
		}
		return T(SigmoidActivation(float64(value), slope))
	case SOFTMAX:
		return T(softmaxActivation(float64(value)))
	case SWISH:
		beta := 1.0
		if len(params) > 0 {
			beta = params[0]
		}
		return T(swishActivation(value, beta))
	case TANH:
		return T(tanhActivation(float64(value)))
	default:
		return value // Default to linear if unknown
	}
}

// Derivative activation function with parameters.
func Derivative[T utils.Float](value T, mode ActivationType, params ...float64) T {
	switch mode {
	case ELISH:
		return T(elishDerivative(float64(value)))
	case ELU:
		alpha := 1.0
		if len(params) > 0 {
			alpha = params[0]
		}
		return T(eluDerivative(value, alpha))
	case LINEAR:
		slope := 1.0
		if len(params) > 0 {
			slope = params[0]
		}
		return T(linearDerivative[float64](slope))
	case LEAKYRELU:
		leak := 0.01
		if len(params) > 0 {
			leak = params[0]
		}
		return T(reluDerivative(value, leak))
	case RELU:
		return T(reluDerivative(value, 0.0))
	case SELU:
		scale := 1.0507009873554804934193349852946
		alpha := 1.673263242354372848170429916717
		if len(params) > 0 {
			scale = params[0]
		}
		if len(params) > 1 {
			alpha = params[1]
		}
		return T(seluDerivative(value, scale, alpha))
	case SIGMOID:
		slope := 1.0
		if len(params) > 0 {
			slope = params[0]
		}
		return T(SigmoidDerivative(float64(value), slope))
	case SOFTMAX:
		return T(softmaxDerivative(float64(value)))
	case SWISH:
		beta := 1.0
		if len(params) > 0 {
			beta = params[0]
		}
		return T(swishDerivative(value, beta))
	case TANH:
		return T(tanhDerivative(float64(value)))
	default:
		return 1.0 // Default derivative if unknown
	}
}
