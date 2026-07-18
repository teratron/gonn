package activation

import (
	"github.com/teratron/gonn/pkg/utils"
)

// ActivationType represents different activation functions.
//
// AI-Meta:
//   - Purpose: Enum selecting which activation function to apply in Activation and Derivative dispatchers.
//   - Usage: Pass as mode argument — activation.Activation[float32](x, activation.SIGMOID).
//   - Related: [Activation], [Derivative], [Function].
type Type uint8

// Activation function mode.
const (
	ELISH     Type = iota // ELISH - Exponential Linear Unit + Sigmoid.
	ELU                   // ELU - Exponential Linear Unit.
	Linear                // Linear - Linear/identity.
	LeakyReLU             // LeakyReLU - Leaky ReLU (leaky rectified linear unit).
	ReLU                  // ReLU - ReLU (rectified linear unit).
	SELU                  // SELU - Scaled Exponential Linear Unit.
	SIGMOID               // SIGMOID - Logistic, a.k.a. sigmoid or soft step.
	SOFTMAX               // SOFTMAX - Softmax (Note: Current implementation is a placeholder for single values, requires vector for full functionality).
	SWISH                 // SWISH - Swish-function.
	TanH                  // TanH - TanH (hyperbolic tangent).
	Default   = Linear
)

// Function is the contract for stateful, in-place activation functions that carry their own hyperparameters.
// Unlike the Activation/Derivative dispatchers, implementations hold state (e.g. slope, alpha).
//
// AI-Meta:
//   - Purpose: Contract for stateful activation functions with configurable hyperparameters.
//   - Usage: s := activation.NewSigmoid[float32](1.0); s.Activation(&v); s.Derivative(&v).
//   - Implementations: [Sigmoid].
//   - Related: [Activation], [Derivative].
type Function[T utils.Float] interface {
	Activation(value *T)
	Derivative(value *T)
}

// Activation applies the named activation function to a single value, dispatching by mode.
// Optional params configure mode-specific hyperparameters (slope, alpha, scale, leak, beta).
//
// AI-Meta:
//   - Purpose: Stateless dispatcher — apply one activation function by mode to a scalar value.
//   - Usage: y := activation.Activation[float32](x, activation.SIGMOID) — params optional per mode.
//   - Related: [Derivative], [Type], [Function].
func Activation[T utils.Float](value T, mode Type, params ...float64) T {
	switch mode {
	case ELISH:
		return T(elishActivation(float64(value)))
	case ELU:
		alpha := 1.0
		if len(params) > 0 {
			alpha = params[0]
		}
		return T(eluActivation(value, alpha))
	case Linear:
		slope := 1.0
		offset := 0.0
		if len(params) > 0 {
			slope = params[0]
		}
		if len(params) > 1 {
			offset = params[1]
		}
		return T(linearActivation(value, slope, offset))
	case LeakyReLU:
		leak := 0.01
		if len(params) > 0 {
			leak = params[0]
		}
		return T(reluActivation(value, leak))
	case ReLU:
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
	case TanH:
		return T(tanhActivation(float64(value)))
	default:
		return value // Default to linear if unknown
	}
}

// Derivative applies the derivative of the named activation function, dispatching by mode.
// For most modes the input value is expected to be the post-activation output, not the raw input.
//
// AI-Meta:
//   - Purpose: Stateless dispatcher — compute one activation derivative by mode for a scalar value.
//   - Usage: d := activation.Derivative[float32](y, activation.SIGMOID) — y is post-activation.
//   - Related: [Activation], [Type], [Function].
func Derivative[T utils.Float](value T, mode Type, params ...float64) T {
	switch mode {
	case ELISH:
		return T(elishDerivative(float64(value)))
	case ELU:
		alpha := 1.0
		if len(params) > 0 {
			alpha = params[0]
		}
		return T(eluDerivative(value, alpha))
	case Linear:
		slope := 1.0
		if len(params) > 0 {
			slope = params[0]
		}
		return T(linearDerivative[float64](slope))
	case LeakyReLU:
		leak := 0.01
		if len(params) > 0 {
			leak = params[0]
		}
		return T(reluDerivative(value, leak))
	case ReLU:
		return T(reluDerivative(value, 0.0))
	case SELU:
		scale := 1.0507009873554804934193349852946
		alpha := 1.6732632423543772848170429916717
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
	case TanH:
		return T(tanhDerivative(float64(value)))
	default:
		return 1.0 // Default derivative if unknown
	}
}

// String returns the string representation of the activation type.
//
// AI-Meta:
//   - Purpose: Human-readable name for the activation type, used in logs and diagnostics.
//   - Usage: fmt.Println(activation.SIGMOID.String()) // → "SIGMOID".
//   - Related: [Type].
func (a Type) String() string {
	switch a {
	case ELISH:
		return "ELISH"
	case ELU:
		return "ELU"
	case Linear:
		return "Linear"
	case LeakyReLU:
		return "LeakyReLU"
	case ReLU:
		return "ReLU"
	case SELU:
		return "SELU"
	case SIGMOID:
		return "SIGMOID"
	case SOFTMAX:
		return "SOFTMAX"
	case SWISH:
		return "SWISH"
	case TanH:
		return "TanH"
	default:
		return "Unknown"
	}
}
