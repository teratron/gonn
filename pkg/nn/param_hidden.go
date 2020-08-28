// Hidden layers
package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

type HiddenType []uint

func HiddenLayer(nums ...uint) HiddenType {
	return nums
}

func (n *NN) HiddenLayer() []uint {
	return n.Architecture.(Parameter).HiddenLayer()
}

// Setter
func (h HiddenType) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(*NN); ok && !n.IsInit {
			n.Get().Set(h)
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Getter
func (h HiddenType) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(h)
		}
	} else {
		return h
	}
	return nil
}
