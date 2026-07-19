package activation

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

var _ Function[float32] = (*Sigmoid[float32])(nil)
var _ Function[float64] = (*Sigmoid[float64])(nil)

// Sigmoid represents the sigmoid activation function with a configurable slope parameter.
// It implements [Function] with in-place Activation and Derivative operations.
//
// AI-Meta:
//   - Purpose: Stateful sigmoid activation carrying a slope hyperparameter; implements Function.
//   - Usage: s := activation.NewSigmoid[float32](1.0); s.Activation(&v); s.Derivative(&v).
//   - Concurrency: NotSafe; modifies value in-place — one instance per layer.
//   - Related: [NewSigmoid], [SigmoidActivation], [SigmoidDerivative], [Function].
type Sigmoid[T utils.Float] struct {
	slope T
}

// NewSigmoid creates a new Sigmoid activation function with the given slope.
//
// AI-Meta:
//   - Purpose: Construct a Sigmoid with the given slope ready to attach to a layer or cell.
//   - Usage: s := activation.NewSigmoid[float32](1.0).
//   - Related: [Sigmoid], [Function].
func NewSigmoid[T utils.Float](slope T) *Sigmoid[T] {
	return &Sigmoid[T]{slope}
}

// Activation applies the sigmoid activation function in-place: *value = 1 / (1 + exp(-slope * *value)).
//
// AI-Meta:
//   - Purpose: Apply sigmoid in-place; overwrites *value with the activation output.
//   - Concurrency: NotSafe; modifies *value.
//   - Related: [Derivative], [SigmoidActivation].
func (s *Sigmoid[T]) Activation(value *T) {
	*value = 1.0 / (1.0 + T(math.Exp(float64(-s.slope**value))))
}

// Derivative calculates the sigmoid derivative in-place; *value must be the post-activation output.
//
// AI-Meta:
//   - Purpose: Apply sigmoid derivative in-place; *value must already hold the activation output.
//   - Concurrency: NotSafe; modifies *value.
//   - Related: [Activation], [SigmoidDerivative].
func (s *Sigmoid[T]) Derivative(value *T) {
	//sigmoidValue := s.Activation(value)
	*value = s.slope * *value * (T(1.0) - *value)
}

// SigmoidActivation computes the sigmoid activation without state: 1 / (1 + exp(-slope * value)).
//
// AI-Meta:
//   - Purpose: Stateless sigmoid activation; returns the scalar output without modifying input.
//   - Usage: y := activation.SigmoidActivation[float32](x, 1.0).
//   - Related: [SigmoidDerivative], [Activation].
func SigmoidActivation[T utils.Float](value T, slope T) T {
	return T(1.0) / (T(1.0) + T(math.Exp(float64(-slope*value))))
}

// SigmoidDerivative computes the sigmoid gradient: slope * σ(x) * (1 - σ(x)).
//
// AI-Meta:
//   - Purpose: Stateless sigmoid derivative; computes gradient from the raw input value.
//   - Usage: d := activation.SigmoidDerivative[float32](x, 1.0).
//   - Related: [SigmoidActivation], [Derivative].
func SigmoidDerivative[T utils.Float](value T, slope T) T {
	sigmoidVal := SigmoidActivation(value, slope)
	return slope * sigmoidVal * (T(1.0) - sigmoidVal)
}
