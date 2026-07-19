package loss

import "github.com/teratron/gonn/pkg/utils"

// Mean Absolute Error loss function: MAE = |predicted - target|
func maeLoss[T utils.Float](predicted, target T) T {
	diff := predicted - target
	if diff < 0 {
		return -diff
	}
	return diff
}
