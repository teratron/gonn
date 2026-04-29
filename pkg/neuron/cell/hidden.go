package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

// Hidden is the interior cell type. Functionally identical to Dense — the
// only distinction is the kind tag stored in Id[0] (neuron.DENSE for the
// generic flavour, kept the same here to avoid introducing a new const
// without spec authority). The alias keeps every Dense method available
// on Hidden values without duplication, while the dedicated constructor
// gives callers a stable name to reach for.
//
// Generic type aliases require Go 1.24+ (this module is on 1.26.2).
type Hidden[T utils.Float] = Dense[T]

var (
	_ neuron.Neuron[float32] = (*Hidden[float32])(nil)
	_ neuron.Neuron[float64] = (*Hidden[float64])(nil)
)

// NewHidden allocates a Hidden cell at position number. Returned as
// *Hidden[T] (= *Dense[T]) so call sites that import only this file can
// stay unaware of the alias.
func NewHidden[T utils.Float](number uint) *Hidden[T] {
	return NewDense[T](number)
}
