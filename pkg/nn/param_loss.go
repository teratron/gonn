package nn

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg"
)

type (
	lossModeUint   uint8   // Average error mode
	lossLevelFloat float64 // Level loss
)

const (
	ModeMSE      uint8   = iota   // Mean Squared Error
	ModeRMSE                      // Root Mean Squared Error
	ModeARCTAN                    // Arctan
	MinLossLevel float64 = 10e-33 // The minimum value of the error limit at which training is forcibly terminated
)

// LossMode
func LossMode(mode ...uint8) pkg.GetSetter {
	if len(mode) > 0 {
		return lossModeUint(mode[0])
	} else {
		return lossModeUint(0)
	}
}

// LossLevel
func LossLevel(level ...float64) pkg.GetSetter {
	if len(level) > 0 {
		return lossLevelFloat(level[0])
	} else {
		return lossLevelFloat(0)
	}
}

// LossMode
func (n *nn) LossMode() uint8 {
	return n.Architecture.(Parameter).LossMode()
}

// LossLevel
func (n *nn) LossLevel() float64 {
	return n.Architecture.(Parameter).LossLevel()
}

// Set
func (m lossModeUint) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(Architecture); ok {
			a.Get().Set(m.check())
		}
	} else {
		errNN(fmt.Errorf("%w set for loss mode", ErrEmpty))
	}
}

// Set
func (l lossLevelFloat) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(Architecture); ok {
			a.Get().Set(l.check())
		}
	} else {
		errNN(fmt.Errorf("%w set for loss level", ErrEmpty))
	}
}

// Get
func (m lossModeUint) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(Architecture); ok {
			return a.Get().Get(m)
		}
	} else {
		return m
	}
	return nil
}

// Get
func (l lossLevelFloat) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(Architecture); ok {
			return a.Get().Get(l)
		}
	} else {
		return l
	}
	return nil
}

// check
func (m lossModeUint) check() lossModeUint {
	switch {
	case m < 0 || m > lossModeUint(ModeARCTAN):
		return lossModeUint(ModeMSE)
	default:
		return m
	}
}

// check
func (l lossLevelFloat) check() lossLevelFloat {
	switch {
	case l < 0 || l < lossLevelFloat(MinLossLevel):
		return lossLevelFloat(MinLossLevel)
	default:
		return l
	}
}
