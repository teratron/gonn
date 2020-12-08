package nn

import "fmt"

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
func (h HiddenArrUint) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok && !n.stateInit() {
			n.Set(h)
		}
	} else {
		LogError(fmt.Errorf("%w set for hidden layers array", ErrEmpty))
	}
}

// Get
func (h HiddenArrUint) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get(h)
		}
	} else {
		return h
	}
	return nil
}
