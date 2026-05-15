package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// base extends core with activation and bias — the two properties that
// every working (non-input) layer needs. Embeds *core[T, S] by pointer
// so concrete layer types share one core block; constructors must build
// the core via newCore before the base is usable.
type base[T utils.Float, S neuron.Neuron[T]] struct {
	*core[T, S]
	bias       *cell.Bias[T]
	Bias       bool            `json:"bias" xml:"bias"`
	Activation activation.Type `json:"activation" xml:"activation"`
}

// newBase allocates a base layer with a fresh core of the given kind. If
// bias is requested the bias cell is created here; otherwise the field
// stays nil and downstream code must check before reading.
func newBase[T utils.Float, S neuron.Neuron[T]](kind uint8, size int, act activation.Type, useBias bool) *base[T, S] {
	b := &base[T, S]{
		core:       newCore[T, S](kind, size),
		Bias:       useBias,
		Activation: act,
	}
	if useBias {
		b.bias = cell.NewBias[T]()
	}
	return b
}

// Init resets a base layer in place. Delegates the size + cell allocation
// to embedded core.Init, then sets the activation and bias fields. The
// bias cell is reallocated if requested — Init is meant to be safe to
// call repeatedly during dynamic-topology adjustments.
func (b *base[T, S]) Init(kind uint8, size int, act activation.Type, useBias bool) {
	if b.core == nil {
		b.core = newCore[T, S](kind, size)
	} else {
		b.core.Init(kind, size)
	}
	b.Activation = act
	b.Bias = useBias
	if useBias {
		b.bias = cell.NewBias[T]()
	} else {
		b.bias = nil
	}
}

// BiasCell returns the bias unit owned by this layer or nil when bias is
// disabled. Callers wiring axons inspect the result to decide whether to
// add the +1 connection.
func (b *base[T, S]) BiasCell() *cell.Bias[T] {
	return b.bias
}
