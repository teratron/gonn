package nn

import "math"

const (
	LINEAR	  uint8 = 0 // Linear/Identity
	SIGMOID   uint8 = 1 // Logistic, a.k.a. sigmoid or soft step
	TANH      uint8 = 2 // TanH - hyperbolic
	RELU      uint8 = 3 // ReLu - rectified linear unit
	LEAKYRELU uint8 = 4 // Leaky ReLu - leaky rectified linear unit
)

type Activator interface {
	Get(float64) float64
}

type Activation struct {
	Mode uint8
}

type Derivative struct {
	Mode uint8
}

func GetActivation(value float64, a Activator) float64 {
	return a.Get(value)
}

//+-------------------------------------------------------------+
//| Activation function                                         |
//+-------------------------------------------------------------+
func (a *Activation) Get(value float64) float64 {
	switch a.Mode {
	default:
		fallthrough
	case LINEAR:
		return value
	case SIGMOID:
		return 1 / (1 + math.Exp(-value))
	case TANH:
		value = math.Exp(2 * value)
		if math.IsInf(value, 1) {
			return 1
		}
		return (value - 1) / (value + 1)
	case RELU:
		switch {
		case value < 0:
			return 0
		default:
			return value
		}
	case LEAKYRELU:
		switch {
		case value < 0:
			return .01 * value
		default:
			return value
		}
	}
}

//+-------------------------------------------------------------+
//| Derivative activation function                              |
//+-------------------------------------------------------------+
func (d *Derivative) Get(value float64) float64 {
	switch d.Mode {
	default:
		fallthrough
	case LINEAR:
		return 1
	case SIGMOID:
		return value * (1 - value)
	case TANH:
		return 1 - math.Pow(value, 2)
	case RELU:
		switch {
		case value < 0:
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
