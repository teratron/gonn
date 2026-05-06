package loss

import (
	"github.com/teratron/gonn/pkg/utils"
	"math"
)

// Arctan Error loss function: ARCTAN = arctan(predicted - target)
func arctanLoss[T utils.Float](predicted, target T) T {
	return T(math.Atan(float64(predicted - target)))
}
