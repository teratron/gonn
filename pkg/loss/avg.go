package loss

import "github.com/teratron/gonn/pkg/utils"

// Average Error (Mean Absolute Error) loss function: AVG = |predicted - target|
func avgLoss[T utils.Float](predicted, target T) T {
	return maeLoss(predicted, target)
}
