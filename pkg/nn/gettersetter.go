package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

/*type GetterSetter interface {
	Getter
	Setter
}

type Getter interface {
	pkg.Getter
}

type Setter interface {
	pkg.Setter
}*/

// Set
func (n *NN) Set(args ...pkg.Setter) {
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

// Get
func (n *NN) Get(args ...pkg.Getter) pkg.GetterSetter {
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