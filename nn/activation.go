package nn

import "math"

const (
	IDENTITY  uint8 = 0 // Identity
	SIGMOID   uint8 = 1 // Logistic, a.k.a. sigmoid or soft step
	TANH      uint8 = 2 // TanH - hyperbolic
	RELU      uint8 = 3 // ReLu - rectified linear unit
	LEAKYRELU uint8 = 4 // Leaky ReLu - leaky rectified linear unit
)

type Activator interface {
	Get() float32
}

type Activation struct {
	Value float32
	Mode  uint8
}

type Derivative struct {
	Value float32
	Mode  uint8
}

// Activation function
func (a *Activation) Get() float32 {
	switch a.Mode {
	default:
		fallthrough
	case IDENTITY:
		return a.Value
	case SIGMOID:
		return float32(1 / (1 + math.Pow(math.E, float64(-a.Value))))
	case TANH:
		a.Value = float32(math.Pow(math.E, float64(2*a.Value)))
		return (a.Value - 1) / (a.Value + 1)
	case RELU:
		switch {
		case a.Value < 0:
			return 0
		case a.Value > 1:
			return 1
		default:
			return a.Value
		}
	case LEAKYRELU:
		switch {
		case a.Value < 0:
			return .01 * a.Value
		case a.Value > 1:
			return 1 + .01*(a.Value-1)
		default:
			return a.Value
		}
	}
}

// Derivative activation function
func (d *Derivative) Get() float32 {
	switch d.Mode {
	default:
		fallthrough
	case IDENTITY:
		return 1
	case SIGMOID:
		return d.Value * (1 - d.Value)
	case TANH:
		return 1 - float32(math.Pow(float64(d.Value), 2))
	case RELU:
		switch {
		case d.Value <= 0:
			return 0
		default:
			return 1
		}
	case LEAKYRELU:
		switch {
		case d.Value < 0:
			return .01
		default:
			return 1
		}
	}
}
