package nn

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg"
)

type biasBool bool

// NeuronBias
func NeuronBias(bias ...bool) pkg.GetSetter {
	if len(bias) > 0 {
		return biasBool(bias[0])
	}
	return biasBool(false)
}

// NeuronBias
func (n *nn) NeuronBias() bool {
	return n.Architecture.(Parameter).NeuronBias()
}

// Set
func (b biasBool) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(*nn); ok && !n.isInit {
			n.Get().Set(b)
		}
	} else {
		errNN(fmt.Errorf("%w set for bias", ErrEmpty))
	}
}

// Get
func (b biasBool) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(Architecture); ok {
			return n.Get().Get(b)
		}
	} else {
		return b
	}
	return nil
}
