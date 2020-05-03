package nn

import "math"

const (
	IDENTITY  uint8 = 0 // Identity (тождественная)
	SIGMOID   uint8 = 1 // Logistic, a.k.a. sigmoid or soft step (логистическая, сигмоида или гладкая ступенька)
	TANH      uint8 = 2 // TanH - hyperbolic (гиперболический тангенс)
	RELU      uint8 = 3 // ReLu - rectified linear unit (линейный выпрямитель)
	LEAKYRELU uint8 = 4 // Leaky ReLu - leaky rectified linear unit (линейный выпрямитель с «утечкой»)
)

// Activation function
func GetActivation(value float32, mode uint8) float32 {
	switch mode {
	default:
		fallthrough
	case IDENTITY:
		return value
	case SIGMOID:
		return float32(1 / (1 + math.Pow(math.E, float64(-value))))
	case TANH:
		value = float32(math.Pow(math.E, float64(2*value)))
		return (value - 1) / (value + 1)
	case RELU:
		switch {
		case value < 0:
			return 0
		case value > 1:
			return 1
		default:
			return value
		}
	case LEAKYRELU:
		switch {
		case value < 0:
			return .01 * value
		case value > 1:
			return 1 + .01*(value-1)
		default:
			return value
		}
	}
}

// Derivative activation function
func GetDerivative(value float32, mode uint8) float32 {
	switch mode {
	default:
		fallthrough
	case IDENTITY:
		return 1
	case SIGMOID:
		return value * (1 - value)
	case TANH:
		return 1 - float32(math.Pow(float64(value), 2))
	case RELU:
		switch {
		case value <= 0:
			return 0
		default:
			return 1
		}
	case LEAKYRELU:
		switch {
		case value < 0:
			return .01
		default:
			return 1
		}
	}
}
