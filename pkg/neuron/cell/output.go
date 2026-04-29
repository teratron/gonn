package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var (
	_ neuron.Neuron[float32] = (*Output[float32])(nil)
	_ neuron.Neuron[float64] = (*Output[float64])(nil)
)

// Output is the terminal cell type — a Dense unit augmented with a target
// pointer used to compute the residual `target - value` after each forward
// pass. The target is owned by the loss layer; Output stores the pointer
// so callers can swap in a new label without re-walking the graph.
type Output[T utils.Float] struct {
	*Dense[T]
	target *T
}

// NewOutput allocates an Output cell pointing at the supplied target slot.
// The Dense backing is created with kind tag neuron.OUTPUT so introspection
// distinguishes terminal cells from interior ones.
func NewOutput[T utils.Float](target *T) *Output[T] {
	return &Output[T]{
		Dense:  NewDense[T](uint(neuron.OUTPUT)),
		target: target,
	}
}

// GetTarget returns the externally owned target pointer. Caller should
// treat it read-only — Output does not own the lifetime of the target.
func (o *Output[T]) GetTarget() *T {
	return o.target
}

// SetTarget repoints the cell at a new target. Used between training
// samples to feed the next label without reallocating Output.
func (o *Output[T]) SetTarget(value *T) {
	o.target = value
}

// CalculateValue (FORWARD) delegates to the embedded Dense's forward pass
// and then writes the residual `target - value` into miss so the backward
// pass starts with a populated error term.
//
// The explicit `o.Dense.CalculateValue()` call is mandatory: a bare
// `o.CalculateValue()` resolves back to this very method via promotion
// and produces unbounded recursion. This was the third half of blocker
// C-001 in the legacy code.
func (o *Output[T]) CalculateValue() {
	o.Dense.CalculateValue()
	if o.target != nil {
		o.SetMiss(*o.target - o.value)
	}
}
