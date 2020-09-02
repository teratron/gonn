package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

// Set
func (n *nn) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		for _, v := range args {
			v.Set(n)
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Get
func (n *nn) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		for _, v := range args {
			return v.Get(n)
		}
	} else {
		if a, ok := n.Architecture.(NeuralNetwork); ok {
			return a
		}
	}
	return n
}
