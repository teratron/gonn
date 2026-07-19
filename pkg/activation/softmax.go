package activation

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// softmaxActivation applies a per-element sigmoid used as a scalar stand-in for
// a true vector softmax. Uses the numerically stable form to prevent exp(+large)
// from overflowing float32/float64 to +Inf and producing NaN.
func softmaxActivation[T utils.Float](value T) T {
	if value >= 0 {
		return T(1.0) / (T(1.0) + T(math.Exp(-float64(value))))
	}
	expVal := T(math.Exp(float64(value)))
	return expVal / (expVal + T(1.0))
}

// softmaxDerivative returns σ(x)·(1−σ(x)), the derivative of the scalar sigmoid.
func softmaxDerivative[T utils.Float](value T) T {
	a := softmaxActivation(value)
	return a * (T(1.0) - a)
}
