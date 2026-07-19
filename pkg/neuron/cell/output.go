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
//
// AI-Meta:
//   - Purpose: Terminal cell that computes loss residual and drives the backward pass.
//   - Concurrency: NotSafe; CalculateValue writes miss in-place.
//   - Related: [NewOutput], [Dense], [neuron.Neuron].
type Output[T utils.Float] struct {
	*Dense[T]
	target *T
}

// NewOutput allocates an Output cell pointing at the supplied target slot.
// The Dense backing is created with kind tag neuron.OUTPUT so introspection
// distinguishes terminal cells from interior ones.
//
// AI-Meta:
//   - Purpose: Construct an Output cell wired to an external target slot; used by layer.NewOutput.
//   - Usage: c := cell.NewOutput[float32](&targetSlot).
//   - Related: [Output].
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

// CalculateValue (FORWARD) delegates to Dense's forward pass then computes
// miss = target - value. The explicit Dense.CalculateValue() call is
// mandatory to avoid infinite recursion via method promotion.
//
// AI-Meta:
//   - Purpose: Forward pass for output cell: compute dot product then write residual into miss.
//   - Concurrency: NotSafe; mutates miss and value.
//   - Related: [Dense.CalculateValue], [Dense.CalculateWeight].
func (o *Output[T]) CalculateValue() {
	o.Dense.CalculateValue()
	if o.target != nil {
		o.SetMiss(*o.target - o.value)
	}
}
