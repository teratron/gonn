package activation

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

type ActivationFunction[T float32 | float64] interface {
	activation(value T) T
	derivative(value T) T
}

// Activation function with parameters.
func Activation[T float32 | float64](value T, mode ActivationType, params ...float64) T {
	switch mode {
	case ELISH:
		return elishActivation(value)
	case ELU:
		alpha := 1.0
		if len(params) > 0 {
			alpha = params[0]
		}
		return eluActivation(value, alpha)
	case LINEAR:
		slope := 1.0
		offset := 0.0
		if len(params) > 0 {
			slope = params[0]
		}
		if len(params) > 1 {
			offset = params[1]
		}
		return linearActivation(value, slope, offset)
	case LEAKYRELU:
		leak := 0.01
		if len(params) > 0 {
			leak = params[0]
		}
		return reluActivation(value, leak)
	case RELU:
		return reluActivation(value, 0.0)
	case SELU:
		scale := 1.0507009873554804934193349852946
		alpha := 1.6732632423543772848170429916717
		if len(params) > 0 {
			scale = params[0]
		}
		if len(params) > 1 {
			alpha = params[1]
		}
		return seluActivation(value, scale, alpha)
	case SIGMOID:
		slope := 1.0
		if len(params) > 0 {
			slope = params[0]
		}
		return sigmoidActivation(value, slope)
	case SOFTMAX:
		return softmaxActivation(value)
	case SWISH:
		beta := 1.0
		if len(params) > 0 {
			beta = params[0]
		}
		return swishActivation(value, beta)
	case TANH:
		return tanhActivation(value)
	default:
		return value // Default to linear if unknown
	}
}

// Derivative activation function with parameters.
func Derivative[T float32 | float64](value T, mode ActivationType, params ...float64) T {
	switch mode {
	case ELISH:
		return elishDerivative(value)
	case ELU:
		alpha := 1.0
		if len(params) > 0 {
			alpha = params[0]
		}
		return eluDerivative(value, alpha)
	case LINEAR:
		slope := 1.0
		if len(params) > 0 {
			slope = params[0]
		}
		return T(linearDerivative(slope))
	case LEAKYRELU:
		leak := 0.01
		if len(params) > 0 {
			leak = params[0]
		}
		return reluDerivative(value, leak)
	case RELU:
		return reluDerivative(value, 0.0)
	case SELU:
		scale := 1.0507009873554804934193349852946
		alpha := 1.673263242354372848170429916717
		if len(params) > 0 {
			scale = params[0]
		}
		if len(params) > 1 {
			alpha = params[1]
		}
		return seluDerivative(value, scale, alpha)
	case SIGMOID:
		slope := 1.0
		if len(params) > 0 {
			slope = params[0]
		}
		return sigmoidDerivative(value, slope)
	case SOFTMAX:
		return softmaxDerivative(value)
	case SWISH:
		beta := 1.0
		if len(params) > 0 {
			beta = params[0]
		}
		return swishDerivative(value, beta)
	case TANH:
		return tanhDerivative(value)
	default:
		return 1.0 // Default derivative if unknown
	}
}
