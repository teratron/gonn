package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/utils"
)

var (
	_ neuron.Neuron[float32] = (*Dense[float32])(nil)
	_ neuron.Neuron[float64] = (*Dense[float64])(nil)
)

// Dense is the work-horse cell type — a fully connected unit that owns a
// bundle of incoming axons, a scalar value (inherited from core via
// embedding), and a backprop error term. Dense satisfies [neuron.Neuron]:
// it can compute its own value from the axon sum and update axon weights
// from a propagated gradient.
//
// AI-Meta:
//   - Purpose: Hidden-layer cell that performs forward (dot product) and backward (weight update) steps.
//   - Concurrency: NotSafe; CalculateValue and CalculateWeight mutate internal state.
//   - Related: [NewDense], [neuron.Neuron], [axon.Bundle].
type Dense[T utils.Float] struct {
	*core[T]
	miss  T
	Axons axon.Bundle[T] `json:"axons" xml:"axons"`
}

// NewDense allocates a Dense cell at position number within its layer.
// The axon bundle starts empty — connections are wired in by the layer
// constructor once both endpoints exist.
//
// AI-Meta:
//   - Purpose: Construct a Dense cell with an empty axon bundle; used by layer.NewDense.
//   - Usage: c := cell.NewDense[float32](0).
//   - Related: [Dense].
func NewDense[T utils.Float](number uint) *Dense[T] {
	return &Dense[T]{
		core:  newCore[T]([2]uint{uint(neuron.DENSE), number}),
		miss:  0,
		Axons: make(axon.Bundle[T], 0),
	}
}

// GetMiss returns a pointer to the accumulated backprop error. Mirrors
// the GetValue pattern — pointer stays stable for the cell's lifetime.
func (d *Dense[T]) GetMiss() *T {
	return &d.miss
}

// SetMiss overwrites the error term. Used at the start of each backward
// pass to reset the accumulator before fan-in.
func (d *Dense[T]) SetMiss(value T) {
	d.miss = value
}

// AddMiss accumulates a partial error contribution onto the existing miss.
// Backprop sums one term per outgoing axon; AddMiss is the per-term hook.
// Not part of the [neuron.Neuron] interface — callers that work through
// the interface must downcast to *Dense[T] (or its alias *Hidden[T]).
func (d *Dense[T]) AddMiss(value T) {
	d.miss += value
}

// CalculateValue (FORWARD) refreshes the scalar value from the dot product
// of incoming-axon weights and source-cell values. Activation application
// is intentionally absent here — it lands on the layer level.
//
// AI-Meta:
//   - Purpose: Run the forward pass for this cell: value = Σ(axon.Weight * input.Value).
//   - Concurrency: NotSafe; mutates d.value.
//   - Related: [CalculateWeight], [axon.Axon.CalculateValue].
func (d *Dense[T]) CalculateValue() {
	d.value = 0
	for _, a := range d.Axons {
		d.value += a.CalculateValue()
	}
}

// CalculateWeight (BACKWARD) updates every incoming axon weight using the
// supplied learning rate and the cell's accumulated miss. The gradient
// passed to each axon is rate * miss.
//
// AI-Meta:
//   - Purpose: Run the backward pass: update all incoming axon weights from the accumulated gradient.
//   - Concurrency: NotSafe; mutates axon weights.
//   - Related: [CalculateValue], [axon.Axon.CalculateWeight].
func (d *Dense[T]) CalculateWeight(rate *T) {
	gradient := *rate * d.miss
	for i := range d.Axons {
		d.Axons[i].CalculateWeight(&gradient)
	}
}
