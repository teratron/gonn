package activation

import (
	"github.com/teratron/gonn/pkg/utils"
	"math"
)

// TanH activation function: f(x) = tanh(x)
func tanhActivation[T utils.Float](value T) T {
	return T(math.Tanh(float64(value)))
}

// TanH derivative function: f'(x) = 1 - tanh(x)^2.
// The dispatcher convention is pre-activation input (value is the raw preact
// z, not tanh(z)), so tanh(z) is recomputed here — consistent with every other
// activation in this package.
func tanhDerivative[T utils.Float](value T) T {
	t := T(math.Tanh(float64(value)))
	return T(1.0) - t*t
}
