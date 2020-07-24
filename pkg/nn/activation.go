//
package nn

import "math"

type modeActivationType uint8

const (
	ModeLINEAR uint8 = iota	// Linear/identity
	ModeRELU				// ReLu - rectified linear unit
	ModeLEAKYRELU			// Leaky ReLu - leaky rectified linear unit
	ModeSIGMOID				// Logistic, a.k.a. sigmoid or soft step
	ModeTANH				// TanH - hyperbolic tangent
)

func ModeActivation(mode ...uint8) GetterSetter {
	if len(mode) > 0 {
		return modeActivationType(mode[0])
	} else {
		return modeActivationType(0)
	}
}

// Setter
func (m modeActivationType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(m.check())
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (m modeActivationType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(m)
		}
	} else {
		return m
	}
	return nil
}

// Checking
func (m modeActivationType) check() modeActivationType {
	switch {
	case m < 0 || m > modeActivationType(ModeTANH):
		return modeActivationType(ModeSIGMOID)
	default:
		return m
	}
}

// Activation function
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

// Derivative activation function
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