package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// Output is the terminal layer type. Inherits Activation/Bias from base
// and adds a Loss tag — the loss function symbol applied during training
// (resolved via the loss dispatcher, not stored as a closure here).
//
// AI-Meta:
//   - Purpose: Terminal layer owning Output cells, a target buffer, and the loss type tag.
//   - Concurrency: NotSafe; SetTarget and cell forward pass mutate shared state.
//   - Related: [NewOutput], [cell.Output], [Dense], [loss.Type].
type Output[T utils.Float] struct {
	*base[T, *cell.Output[T]]
	Loss    loss.Type `json:"loss" xml:"loss"`
	targets []T
}

// NewOutput allocates an Output layer of the requested size with the
// chosen activation, loss, and bias settings. Each cell is given a target
// pointer into the layer's internal targets slice — callers update labels
// via SetTarget(idx, value) without re-walking the cells.
//
// AI-Meta:
//   - Purpose: Construct an Output layer with pre-allocated cells and target buffer; used by the network builder.
//   - Usage: l := layer.NewOutput[float32](2, activation.Softmax, loss.CrossEntropy, false).
//   - Related: [Output], [Init], [SetTarget], [cell.NewOutput].
func NewOutput[T utils.Float](size int, act activation.Type, lossKind loss.Type, useBias bool) *Output[T] {
	if size < 0 {
		size = 0
	}
	o := &Output[T]{
		base:    newBase[T, *cell.Output[T]](neuron.OUTPUT, size, act, useBias),
		Loss:    lossKind,
		targets: make([]T, size),
	}
	o.populate()
	utils.Logger.Info("Output layer created",
		"size", size,
		"activation", act.String(),
		"loss", lossKind.String(),
		"bias", useBias,
	)
	return o
}

// Init resets the Output layer to new dimensions and re-allocates the
// targets buffer, delegating activation/bias work to base.Init.
//
// AI-Meta:
//   - Purpose: Resize and re-initialise the Output layer; reuses base or allocates a new one.
//   - Concurrency: NotSafe; must not run during a forward or backward pass.
//   - Related: [NewOutput].
func (o *Output[T]) Init(size int, act activation.Type, lossKind loss.Type, useBias bool) {
	if size < 0 {
		size = 0
	}
	if o.base == nil {
		o.base = newBase[T, *cell.Output[T]](neuron.OUTPUT, size, act, useBias)
	} else {
		o.base.Init(neuron.OUTPUT, size, act, useBias)
	}
	o.Loss = lossKind
	o.targets = make([]T, size)
	o.populate()
}

// populate creates the cell.Output units, each pointing at its own slot
// in the targets slice. The pointers stay valid for the lifetime of the
// layer — SetTarget mutates the slot in place rather than swapping the
// pointer.
func (o *Output[T]) populate() {
	for idx := range o.cells {
		o.cells[idx] = cell.NewOutput[T](&o.targets[idx])
	}
}

// SetTarget writes the label for cell idx. The network drives indices that
// always fit within the targets slice.
//
// AI-Meta:
//   - Purpose: Feed the ground-truth label for one output unit before the backward pass.
//   - Concurrency: NotSafe; must be called before CalculateValue, not concurrently.
//   - Related: [Targets], [cell.Output.SetTarget].
func (o *Output[T]) SetTarget(idx int, value T) {
	o.targets[idx] = value
}

// Targets exposes the label slice. The returned slice shares storage with
// the layer; callers must not modify elements directly — use SetTarget.
//
// AI-Meta:
//   - Purpose: Read the current target vector (e.g. to compute aggregate loss after a forward pass).
//   - Concurrency: ReadSafe after SetTarget completes; do not read during SetTarget.
//   - Related: [SetTarget].
func (o *Output[T]) Targets() []T {
	return o.targets
}

// Cells exposes the output cell slice for axon wiring and loss computation.
//
// AI-Meta:
//   - Purpose: Return the cell slice so the network builder can wire axons and read residuals.
//   - Concurrency: ReadSafe; elements are mutated only by CalculateValue/CalculateWeight.
//   - Related: [cell.Output].
func (o *Output[T]) Cells() []*cell.Output[T] {
	return o.base.Cells()
}
