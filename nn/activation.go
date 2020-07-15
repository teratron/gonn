//
package nn

import "math"

type modeActivationType uint8	// Activation function mode

const (
	ModeLINEAR    modeActivationType = 0	// Linear/identity
	ModeRELU      modeActivationType = 1	// ReLu - rectified linear unit
	ModeLEAKYRELU modeActivationType = 2	// Leaky ReLu - leaky rectified linear unit
	ModeSIGMOID   modeActivationType = 3	// Logistic, a.k.a. sigmoid or soft step
	ModeTANH      modeActivationType = 4	// TanH - hyperbolic tangent
)

func ModeActivation(mode ...modeActivationType) Setter {
	if len(mode) == 0 {
		return modeActivationType(0)
	} else {
		return mode[0]
	}
}

// Setter
func (m modeActivationType) Set(args ...Setter) {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			if c, ok := m.check().(modeActivationType); ok {
				v.Set(c)
			}
		}
	}
}

// Getter
func (m modeActivationType) Get(args ...Setter) Getter {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			return v.Get(m)
		}
	}
	return nil
}

// Checker
func (m modeActivationType) check() Getter {
	switch {
	case m < 0 || m > ModeTANH:
		return ModeSIGMOID
	default:
		return m
	}
}

// Activation function
func calcActivation(value float64, mode modeActivationType) float64 {
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
func calcDerivative(value float64, mode modeActivationType) float64 {
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