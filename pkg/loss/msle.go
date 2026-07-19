package loss

import (
	"github.com/teratron/gonn/pkg/utils"
	"math"
)

// Mean Squared Logarithmic Error loss function: MSLE = (log(predicted + 1) - log(target + 1))²
func msleLoss[T utils.Float](predicted, target T) T {
	logPred := T(math.Log(float64(predicted + 1)))
	logTarget := T(math.Log(float64(target + 1)))
	diff := logPred - logTarget
	return diff * diff
}
