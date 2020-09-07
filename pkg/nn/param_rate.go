package nn

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg"
)

type rateType floatType

// Default learning rate
const DefaultRate float32 = .3

// Rate
func Rate(rate ...float32) pkg.GetSetter {
	if len(rate) > 0 {
		return rateType(rate[0])
	} else {
		return rateType(0)
	}
}

// Rate
func (n *nn) Rate() float32 {
	return n.Architecture.(Parameter).Rate()
}

// Set
func (r rateType) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(Architecture); ok {
			a.Get().Set(r.check())
		}
	} else {
		errNN(fmt.Errorf("%w set for rate", ErrEmpty))
	}
}

// Get
func (r rateType) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(Architecture); ok {
			return a.Get().Get(r)
		}
	} else {
		return r
	}
	return nil
}

// check
func (r rateType) check() rateType {
	switch {
	case r < 0 || r > 1:
		return rateType(DefaultRate)
	default:
		return r
	}
}