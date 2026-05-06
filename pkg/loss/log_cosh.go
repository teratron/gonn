package loss

import (
	"github.com/teratron/gonn/pkg/utils"
	"math"
)

// Log-Cosh Loss: LOG_COSH = log(cosh(predicted - target))
func logCoshLoss[T utils.Float](predicted, target T) T {
	diff := predicted - target
	return T(math.Log(math.Cosh(float64(diff))))
}
