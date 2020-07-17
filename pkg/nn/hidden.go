// Hidden layers
package nn

type (
	hiddenType uint16
	HiddenType []hiddenType
)

func HiddenLayer(nums ...hiddenType) HiddenType {
	return nums
}

// Setter
func (h HiddenType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(h)
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (h HiddenType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(h)
		}
	} else {
		return h
	}
	return nil
}