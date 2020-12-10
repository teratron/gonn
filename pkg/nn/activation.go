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
	switch {
	case mode < 0 || mode > ModeTANH:
		return ModeSIGMOID
	default:
		return mode
	}
}

// calcActivation activation function
func calcActivation(value float64, mode uint8) float64 {
	switch mode {
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

// calcDerivative derivative activation function
func calcDerivative(value float64, mode uint8) float64 {
	switch mode {
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
