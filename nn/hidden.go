// Hidden layers
package nn

type (
	hidden uint16
	Hidden []hidden
)

func HiddenLayer(nums ...hidden) Hidden {
	return nums
}

// Setter
func (h Hidden) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		//if v, ok := n.architecture.(NeuralNetwork); ok {
			n.architecture.Set(h)
		//}
	}
}

// Getter
func (h Hidden) Get(args ...Getter) Getter {
	if n, ok := args[0].(*nn); ok {
		//if v, ok := n.architecture.(NeuralNetwork); ok {
			return n.architecture.Get(h)
		//}
	}
	return nil
}

// Initializing
func (n *nn) SetHidden(args ...hidden) {
	//if v, ok := n.architecture.(NeuralNetwork); ok {
		n.architecture.SetHidden(args...)
	//}
}

// Return
func (n *nn) GetHidden() Hidden {
	//if v, ok := n.architecture.(NeuralNetwork); ok {
		return n.architecture.GetHidden()
	//}
	//return nil
}