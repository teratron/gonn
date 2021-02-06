package nn

import "math"

const (
	// ModeLINEAR - Linear/identity
	ModeLINEAR uint8 = iota

	// ModeRELU - ReLu (rectified linear unit)
	ModeRELU

	// ModeLEAKYRELU - Leaky ReLu (leaky rectified linear unit)
	ModeLEAKYRELU

	// ModeSIGMOID - Logistic, a.k.a. sigmoid or soft step
	ModeSIGMOID

	// ModeTANH - TanH (hyperbolic tangent)
	ModeTANH
)

// checkActivationMode
func checkActivationMode(mode uint8) uint8 {
	if mode > ModeTANH {
		return ModeSIGMOID
	}
	return mode
}

// Activation function
func Activation(value float64, mode uint8) float64 {
	switch mode {
	case ModeLINEAR:
		return value
	case ModeRELU:
		if value < 0 {
			return 0
		}
		return value
	case ModeLEAKYRELU:
		if value < 0 {
			return .01 * value
		}
		return value
	default:
		fallthrough
	case ModeSIGMOID:
		return 1 / (1 + math.Exp(-value))
	case ModeTANH:
		value = math.Exp(2 * value)
		return (value - 1) / (value + 1)
	}
}

// Derivative activation function
func Derivative(value float64, mode uint8) float64 {
	switch mode {
	case ModeLINEAR:
		return 1
	case ModeRELU:
		if value < 0 {
			return 0
		}
		return 1
	case ModeLEAKYRELU:
		if value < 0 {
			return .01
		}
		return 1
	default:
		fallthrough
	case ModeSIGMOID:
		return value * (1 - value)
	case ModeTANH:
		return 1 - math.Pow(value, 2)
	}
}
