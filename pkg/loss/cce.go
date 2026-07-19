package loss

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// cceLossSingle is the per-output contribution to categorical cross-entropy,
// −t·log(y), with y floored to keep the log finite. Summed across the output
// vector by CalculateTotalLoss it yields the standard CCE. (The historical
// implementation returned a constant 0, which made every CCE run report zero
// loss and "converge" at epoch 1.)
func cceLossSingle[T utils.Float](predicted, target T) T {
	p := predicted
	if p < T(1e-7) {
		p = T(1e-7)
	}
	return -target * T(math.Log(float64(p)))
}
