package activation

import (
	"github.com/teratron/gonn/pkg/utils"
	"math"
)

// Swish activation function: f(x, β) = x * sigmoid(β*x) = x / (1 + exp(-β*x))
func swishActivation[T utils.Float](value T, beta float64) T {
	betaVal := T(beta)
	sigmoidBetaX := T(1.0) / (T(math.Exp(float64(-betaVal*value))) + T(1.0))
	return value * sigmoidBetaX
}

// Swish derivative function: f'(x, β) = f(x, β) + sigmoid(β*x) * (1 - f(x, β))
func swishDerivative[T utils.Float](value T, beta float64) T {
	betaVal := T(beta)
	sigmoidBetaX := T(1.0) / (T(math.Exp(float64(-betaVal*value))) + T(1.0))
	swishVal := value * sigmoidBetaX
	return swishVal + sigmoidBetaX*(T(1.0)-swishVal)
}
