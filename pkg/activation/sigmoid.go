package activation

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// Sigmoid represents the sigmoid activation function with a configurable slope parameter
type Sigmoid[T utils.Float] struct {
	slope T
}

// NewSigmoid creates a new Sigmoid activation function with the given slope
func NewSigmoid[T utils.Float](slope T) *Sigmoid[T] {
	return &Sigmoid[T]{slope}
}

// Activation applies the sigmoid activation function: f(x) = 1 / (1 + exp(-slope * x))
func (s *Sigmoid[T]) Activation(value T) T {
	return T(1.0) / (T(1.0) + T(math.Exp(float64(-s.slope*value))))
}

// Derivative calculates the derivative of the sigmoid function
func (s *Sigmoid[T]) Derivative(value T) T {
	sigmoidValue := s.Activation(value)
	return s.slope * sigmoidValue * (T(1.0) - sigmoidValue)
}

// Sigmoid activation function: f(x) = 1 / (1 + exp(-slope * x))
func SigmoidActivation[T utils.Float](value T, slope T) T {
	return T(1.0) / (T(1.0) + T(math.Exp(float64(-slope*value))))
}

// Sigmoid derivative function
func SigmoidDerivative[T utils.Float](value T, slope T) T {
	sigmoidVal := SigmoidActivation(value, slope)
	return slope * sigmoidVal * (T(1.0) - sigmoidVal)
}
