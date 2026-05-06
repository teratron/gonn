package loss

import "github.com/teratron/gonn/pkg/utils"

// Mean Squared Error loss function: MSE = (predicted - target)²
func mseLoss[T utils.Float](predicted, target T) T {
	diff := predicted - target
	return diff * diff
}
