//
package nn

import "math"

const (
	ModeLINEAR    uint8 = 0 // Linear/Identity
	ModeRELU      uint8 = 1 // ReLu - rectified linear unit
	ModeLEAKYRELU uint8 = 2 // Leaky ReLu - leaky rectified linear unit
	ModeSIGMOID   uint8 = 3 // Logistic, a.k.a. sigmoid or soft step
	ModeTANH      uint8 = 4 // TanH - hyperbolic
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
	case ModeLINEAR:
		return value
	case ModeRELU:
		switch {
		case value < 0:
			return 0
		default:
			return value
		}
	case ModeLEAKYRELU:
		switch {
		case value < 0:
			return .01 * value
		default:
			return value
		}
	case ModeSIGMOID:
		return 1 / (1 + math.Exp(-value))
	case ModeTANH:
		value = math.Exp(2 * value)
		if math.IsInf(value, 1) {
			return 1
		}
		return (value - 1) / (value + 1)
	}
}

//+-------------------------------------------------------------+
//| Derivative activation function                              |
//+-------------------------------------------------------------+
func (d *Derivative) Get(value float64) float64 {
	switch d.Mode {
	default:
		fallthrough
	case ModeLINEAR:
		return 1
	case ModeRELU:
		switch {
		case value < 0:
			return 0
		default:
			return 1
		}
	case ModeLEAKYRELU:
		switch {
		case value < 0:
			return .01
		default:
			return 1
		}
	case ModeSIGMOID:
		return value * (1 - value)
	case ModeTANH:
		return 1 - math.Pow(value, 2)
	}
}
