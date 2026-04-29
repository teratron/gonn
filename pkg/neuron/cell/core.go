// Package cell — concrete neural cell types implementing the [neuron.Nucleus]
// and [neuron.Neuron] contracts from [l2-neuron-model].
package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var (
	_ neuron.Nucleus[float32] = (*core[float32])(nil)
	_ neuron.Nucleus[float64] = (*core[float64])(nil)
)

// core is the unexported base for every concrete cell type. It carries the
// minimal state — a 2-tuple identity (layer-kind tag, position index) and a
// scalar value — and satisfies [neuron.Nucleus] via GetValue. Specialised
// cells (Input, Bias, Dense, Output) extend it through Go embedding (C28).
type core[T utils.Float] struct {
	Id    [2]uint `json:"id" xml:"id"`
	value T
}

// newCore returns a freshly allocated core with a zero value. id encodes
// (kind, index) so downstream components can disambiguate cells without a
// pointer-equality check.
func newCore[T utils.Float](id [2]uint) *core[T] {
	return &core[T]{Id: id}
}

// GetValue returns a pointer to the cell's stored scalar. The pointer is
// stable for the lifetime of the cell — callers may cache it for hot-path
// access without re-resolving on every iteration.
func (c *core[T]) GetValue() *T {
	return &c.value
}

// SetValue overwrites the stored scalar. Provided on core itself so that
// every embedding cell type inherits write-access without redeclaring it.
func (c *core[T]) SetValue(value T) {
	c.value = value
}
