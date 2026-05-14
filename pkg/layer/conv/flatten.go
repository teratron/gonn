package conv

import "github.com/teratron/gonn/pkg/utils"

// Flatten is a stateless reshape layer that collapses a feature map into a
// 1-D vector. It carries no weights, no biases, and no learnable state —
// Forward is the identity function (no copy), Backward is the identity
// reshape back to the recorded input shape (CONV-6).
//
// AI-Meta:
//   - Purpose: Stateless 1-D shape-collapse layer between conv stack and Dense head.
//   - Concurrency: NotSafe; mutates inLen on Forward.
//   - Stability: Stable.
//   - Related: [Layer], [Conv1D], [MaxPool1D].
type Flatten[T utils.Float] struct {
	inLen int
}

// NewFlatten constructs a stateless flatten layer.
func NewFlatten[T utils.Float]() *Flatten[T] {
	return &Flatten[T]{}
}

// InputSize / OutputSize / GradSlots — stateless shape-only contract.
func (f *Flatten[T]) InputSize() int                { return f.inLen }
func (f *Flatten[T]) OutputSize() int               { return f.inLen }
func (f *Flatten[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// Forward records the input length and returns the input slice unchanged.
// The output aliases the input — this is intentional and safe because the
// receiver only consumes the slice during the forward sweep.
func (f *Flatten[T]) Forward(x []T) []T {
	f.inLen = len(x)
	return x
}

// Backward returns the upstream gradient unchanged. Length must match the
// recorded input length; mismatches are reported as nil (shape contract).
func (f *Flatten[T]) Backward(upstream []T) []T {
	if len(upstream) != f.inLen {
		return nil
	}
	return upstream
}

// Validate is a no-op — Flatten has no parameters to check.
func (f *Flatten[T]) Validate(inLen int) error {
	_ = inLen
	return nil
}
