package conv

import "github.com/teratron/gonn/pkg/utils"

// Flatten2D is a stateless reshape layer that collapses a CHW feature map
// into a 1-D vector. Element order follows CONV2D-C9 flat-index: element
// (c, y, x) is at output[c*H*W + y*W + x]. The forward pass is the
// identity function (no copy); the backward pass returns the upstream
// gradient unchanged, reusing the recorded shape for downstream layers
// to interpret element order.
//
// AI-Meta:
//   - Purpose: Stateless 2-D shape-collapse layer between Conv2D stack and Dense head.
//   - Concurrency: NotSafe; mutates inC/inH/inW on Forward.
//   - Stability: Stable.
//   - Related: [Layer], [Conv2D], [MaxPool2D], [Flatten].
type Flatten2D[T utils.Float] struct {
	InChannels int `json:"in_channels,omitempty"`
	InH        int `json:"in_h,omitempty"`
	InW        int `json:"in_w,omitempty"`
}

// NewFlatten2D constructs a stateless 2-D flatten layer.
func NewFlatten2D[T utils.Float]() *Flatten2D[T] {
	return &Flatten2D[T]{}
}

// SetInputShape records the spatial dimensions so OutputSize and Backward
// can validate length without invoking Forward first.
func (f *Flatten2D[T]) SetInputShape(inC, inH, inW int) {
	f.InChannels = inC
	f.InH = inH
	f.InW = inW
}

// InputSize / OutputSize — both equal the flat product InChannels*InH*InW.
func (f *Flatten2D[T]) InputSize() int {
	if f.InChannels <= 0 || f.InH <= 0 || f.InW <= 0 {
		return 0
	}
	return f.InChannels * f.InH * f.InW
}

func (f *Flatten2D[T]) OutputSize() int { return f.InputSize() }

// GradSlots returns (nil, nil) — Flatten2D has no trainable parameters.
func (f *Flatten2D[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// OutputShape returns (InChannels, InH, InW). Flatten2D doesn't reshape
// the data, only the upstream layers' interpretation of element order.
func (f *Flatten2D[T]) OutputShape() (c, h, w int) {
	return f.InChannels, f.InH, f.InW
}

// Forward records the input length and returns the input slice unchanged.
// The output aliases the input — this is intentional and safe because the
// receiver only consumes the slice during the forward sweep.
func (f *Flatten2D[T]) Forward(x []T) []T {
	if f.InChannels <= 0 || f.InH <= 0 || f.InW <= 0 {
		// Infer single-channel square shape when feasible.
		s := isqrt(len(x))
		if s*s == len(x) {
			f.InChannels, f.InH, f.InW = 1, s, s
		}
	}
	return x
}

// Backward returns the upstream gradient unchanged. Length must match the
// recorded input length; mismatches are reported as nil (shape contract).
// CONV2D-6: element order MUST follow the flat-index formula
// c*H*W + y*W + x — the upstream gradient is already in that order
// because Flatten2D is pure reshape.
func (f *Flatten2D[T]) Backward(upstream []T) []T {
	if len(upstream) != f.InputSize() {
		return nil
	}
	return upstream
}

// Validate is a no-op — Flatten2D has no parameters to check.
func (f *Flatten2D[T]) Validate(inC, inH, inW int) error {
	_, _, _ = inC, inH, inW
	return nil
}
