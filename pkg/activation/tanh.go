package activation

import (
	"github.com/teratron/gonn/pkg/utils"
	"math"
)

// TanH activation function: f(x) = tanh(x)
func tanhActivation[T utils.Float](value T) T {
	return T(math.Tanh(float64(value)))
}

// TanH derivative function: f'(x) = 1 - tanh(x)^2
func tanhDerivative[T utils.Float](value T) T {
	return T(1.0) - value*value
}
