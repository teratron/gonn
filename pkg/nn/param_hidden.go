package nn

import (
	"fmt"

	"github.com/teratron/gonn/pkg"
)

// HiddenArrUint
type HiddenArrUint []uint

// HiddenLayer
func HiddenLayer(nums ...uint) HiddenArrUint {
	if len(nums) > 0 && nums[0] == 0 {
		return HiddenArrUint{}
	}
	return nums
}

// Set
func (h HiddenArrUint) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		/*if n, ok := args[0].(*nn); ok && !n.isInit {
			n.Get().Set(h)
		}*/
	} else {
		pkg.LogError(fmt.Errorf("%w set for bias", pkg.ErrEmpty))
	}
}

// Get
func (h HiddenArrUint) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(h)
		}
	} else {
		return h
	}
	return nil
}
