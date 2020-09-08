package nn

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg"
)

type HiddenType []uint

// HiddenLayer
func HiddenLayer(nums ...uint) HiddenType {
	return nums
}

// HiddenLayer
func (n *nn) HiddenLayer() []uint {
	return n.Architecture.(Parameter).HiddenLayer()
}

// Set
func (h HiddenType) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(*nn); ok && !n.IsInit {
			n.Get().Set(h)
		}
	} else {
		errNN(fmt.Errorf("%w set for bias", ErrEmpty))
	}
}

// Get
func (h HiddenType) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(Architecture); ok {
			return n.Get().Get(h)
		}
	} else {
		return h
	}
	return nil
}