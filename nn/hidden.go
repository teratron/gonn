// Hidden layers
package nn

type (
	hidden uint16
	Hidden []hidden
)

func HiddenLayer(args ...hidden) Hidden {
	return args
}

// Setter
func (h Hidden) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		if v, ok := n.architecture.(NeuralNetwork); ok {
			v.Set(h)
		}
	}
}

// Getter
func (h Hidden) Get(args ...Getter) Getter {
	if n, ok := args[0].(*nn); ok {
		if v, ok := n.architecture.(NeuralNetwork); ok {
			return v.Get(h)
		}
	}
	return nil
}

// Initializing hidden layers
func (n *nn) SetHidden(args ...hidden) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		v.SetHidden(args...)
	}
}

// Return hidden layers
func (n *nn) GetHidden() Hidden {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		return v.GetHidden()
	}
	return nil
}