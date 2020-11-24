package nn

import (
	"fmt"

	"github.com/teratron/gonn/pkg"
)

type (
	lossModeUint   uint8   // Average error mode
	lossLimitFloat float64 // Level loss
)

const (
	// ModeMSE - Mean Squared Error
	ModeMSE uint8 = iota

	// ModeRMSE - Root Mean Squared Error
	ModeRMSE

	// ModeARCTAN - Arctan
	ModeARCTAN

	// MinLossLimit the minimum value of the error limit at which training is forcibly terminated
	MinLossLimit float64 = 10e-33
)

// LossMode
func LossMode(mode ...uint8) pkg.GetSetter {
	if len(mode) > 0 {
		return lossModeUint(mode[0])
	}
	return lossModeUint(0)
}

// LossLimit
func LossLimit(level ...float64) pkg.GetSetter {
	if len(level) > 0 {
		return lossLimitFloat(level[0])
	}
	return lossLimitFloat(0)
}

// LossMode
/*func (n *nn) LossMode() uint8 {
	return n.Architecture.(Parameter).LossMode()
}

// LossLimit
func (n *nn) LossLimit() float64 {
	return n.Architecture.(Parameter).LossLimit()
}*/

// Set
func (l lossModeUint) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(l.check())
		}
	} else {
		errNN(fmt.Errorf("%w set for loss mode", ErrEmpty))
	}
}

// Set
func (l lossLimitFloat) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(l.check())
		}
	} else {
		errNN(fmt.Errorf("%w set for loss limit", ErrEmpty))
	}
}

// Get
func (l lossModeUint) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(l)
		}
	} else {
		return l
	}
	return nil
}

// Get
func (l lossLimitFloat) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(l)
		}
	} else {
		return l
	}
	return nil
}

// check
func (l lossModeUint) check() lossModeUint {
	switch {
	case l < 0 || l > lossModeUint(ModeARCTAN):
		return lossModeUint(ModeMSE)
	default:
		return l
	}
}

// check
func (l lossLimitFloat) check() lossLimitFloat {
	switch {
	case l < 0 || l < lossLimitFloat(MinLossLimit):
		return lossLimitFloat(MinLossLimit)
	default:
		return l
	}
}
