package loss

import "github.com/teratron/gonn/pkg/utils"

// Squared Hinge Loss: SQ_HINGE = max(0, 1 - predicted * target)²
// This assumes target is -1 or 1, and predicted is the raw output from the model
func sqHingeLoss[T utils.Float](predicted, target T) T {
	margin := 1 - predicted*target
	if margin > 0 {
		return margin * margin
	}
	return 0
}
