//
package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

// Setter
func (n *nn) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		for _, v := range args {
			if s, ok := v.(pkg.Setter); ok {
				s.Set(n)
			}
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Getter
func (n *nn) Get(args ...pkg.Getter) pkg.GetterSetter {
	if len(args) > 0 {
		for _, v := range args {
			if g, ok := v.(pkg.Getter); ok {
				return g.Get(n)
			}
		}
	} else {
		if a, ok := n.Architecture.(NeuralNetwork); ok {
			return a
		}
	}
	return n
}