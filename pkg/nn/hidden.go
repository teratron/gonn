// Hidden layers
package nn

type HiddenType []uint

func HiddenLayer(nums ...uint) HiddenType {
	return nums
}

// Setter
func (h HiddenType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			if n, ok := a.(*NN); ok && !n.isInit {
				a.Get().Set(h)
			}
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