package loss

import "github.com/teratron/gonn/pkg/utils"

// Mean Squared Error loss function: MSE = ½·(predicted - target)².
//
// GoNN uses the ½-scaled form so the gradient is the clean dℓ/dy = (predicted −
// target) with no stray factor of 2. This keeps the training update identical
// to a plain residual step and matches the analytic backprop references in the
// network package. Callers who need the unscaled squared error can double the
// reported value.
func mseLoss[T utils.Float](predicted, target T) T {
	diff := predicted - target
	return 0.5 * diff * diff
}
