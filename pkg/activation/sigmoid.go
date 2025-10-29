package activation

import "math"

// Sigmoid represents the sigmoid activation function with a configurable slope parameter
type Sigmoid[T float32 | float64] struct {
	slope T
}

// NewSigmoid creates a new Sigmoid activation function with the given slope
func NewSigmoid[T float32 | float64](slope T) *Sigmoid[T] {
	return &Sigmoid[T]{slope}
}

// Activation applies the sigmoid activation function: f(x) = 1 / (1 + exp(-slope * x))
func (s *Sigmoid[T]) activation(value T) T {
	return 1.0 / (1.0 + T(math.Exp(float64(-s.slope*value))))
}

// Derivative calculates the derivative of the sigmoid function
func (s *Sigmoid[T]) derivative(value T) T {
	return value * (1.0 - value/s.slope)
}

// Sigmoid activation function: f(x) = slope / (exp(-x) + 1)
func sigmoidActivation[T float32 | float64](value T, slope float64) T {
	return T(slope) / (T(math.Exp(float64(-value))) + T(1.0))
}

// Sigmoid derivative function
func sigmoidDerivative[T float32 | float64](value T, slope float64) T {
	sigmoidVal := sigmoidActivation(value, slope)
	return sigmoidVal * (T(1.0) - sigmoidVal/T(slope))
}
