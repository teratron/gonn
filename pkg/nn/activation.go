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

// Activation activation function
func Activation(value float32, mode uint8) float32 {
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
		return float32(1 / (1 + math.Exp(float64(-value))))
	case ModeTANH:
		value = float32(math.Exp(float64(2 * value)))
		if math.IsInf(float64(value), 1) {
			return 1
		}
		return (value - 1) / (value + 1)
	}
}

// Derivative derivative activation function
func Derivative(value float32, mode uint8) float32 {
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
		return float32(1 - math.Pow(float64(value), 2))
	}
}
