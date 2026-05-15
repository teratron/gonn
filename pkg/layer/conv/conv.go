// Package conv provides 1-D convolutional layer primitives for the GoNN
// neural network library.
//
// Implements the [Layer] interface with three concrete types — [Conv1D],
// [MaxPool1D] / [AvgPool1D], and [Flatten] — satisfying CONV-1..9 from
// l1-conv-layers. Kernel weights participate in the optimizer cycle via
// [Conv1D.GradSlots]; pooling and flatten layers carry no trainable
// parameters (CONV-2).
//
// The package is intentionally decoupled from the neuron-based hierarchy in
// pkg/layer (core/base/cell) because convolutional layers operate on
// kernel-shaped weights, not per-cell weights. The shape contract is exposed
// via [Layer.InputSize] / [Layer.OutputSize] and matches the convention
// established by pkg/layer/norm.
package conv

import (
	"github.com/teratron/gonn/pkg/utils"
)

// PadMode selects the padding strategy for [Conv1D]. PadValid drops samples
// at the boundary so the output is strictly shorter than the input; PadSame
// pads with zeros so the output length matches the input when stride == 1.
//
// AI-Meta:
//   - Purpose: Enum selecting the convolutional padding strategy (Valid vs Same).
//   - Usage: NewConv1D[float32](..., conv.PadValid, ...).
//   - Related: [Conv1D], [outputLen].
//   - Stability: Stable.
type PadMode uint8

const (
	// PadValid drops boundary samples; no zero padding. Output length is
	// floor((inLen - kernelSize) / stride) + 1.
	PadValid PadMode = 0
	// PadSame pads with zeros so the output length is ceil(inLen / stride).
	// Padding is symmetric when (kernelSize - 1) is even, otherwise the
	// extra zero is appended at the right end.
	PadSame PadMode = 1
)

// Layer is the common interface implemented by [Conv1D], [MaxPool1D],
// [AvgPool1D], and [Flatten]. It satisfies the same shape contract as the
// existing layer hierarchy (Dense/Input/Output): a forward call consumes a
// 1-D feature vector and emits a 1-D feature vector of length OutputSize().
//
// Mirrors the pattern from pkg/layer/norm — convolutional layers compose
// alongside the neuron-based layer types via [pkg/nn.Config.ConvPrefix].
//
// AI-Meta:
//   - Purpose: Common interface for 1-D convolutional and reshape layers.
//   - Usage: var _ conv.Layer[float32] = (*conv.Conv1D[float32])(nil).
//   - Implementations: [Conv1D], [MaxPool1D], [AvgPool1D], [Flatten].
//   - Concurrency: NotSafe; Forward / Backward must run from a single goroutine.
//   - Related: [Conv1D], [MaxPool1D], [AvgPool1D], [Flatten].
//   - Stability: Stable.
type Layer[T utils.Float] interface {
	// Forward consumes an input feature vector and returns a freshly allocated
	// output vector of length OutputSize(). The output slice MUST NOT alias
	// the input (CONV-3 forward determinism).
	Forward(x []T) []T
	// Backward consumes the upstream gradient (∂L/∂Y) of length OutputSize()
	// and returns the input-side gradient (∂L/∂X) of length InputSize().
	// Implementations accumulate parameter gradients into internal buffers
	// retrieved via [Layer.GradSlots] (CONV-4).
	Backward(upstream []T) []T
	// InputSize returns the input feature length the layer was constructed
	// for. Returns 0 when the layer is shape-agnostic (e.g. Flatten before
	// the first Forward call).
	InputSize() int
	// OutputSize returns the output feature length after Forward.
	OutputSize() int
	// GradSlots returns the parameter-gradient slices owned by the layer.
	// Returns (nil, nil) when the layer has no trainable parameters
	// (Pool, Flatten). Returns (gradW, gradB) for layers with weights and
	// optional bias.
	GradSlots() (gradW, gradB []T)
}

// outputLen implements CONV-1: the canonical output-length formula for a
// 1-D convolution / pooling operation. Returns 0 when the operation cannot
// produce any output (inLen < kernelSize under PadValid).
//
// Formula:
//   - PadValid: floor((inLen - kernelSize) / stride) + 1.
//   - PadSame:  ceil(inLen / stride).
//
// AI-Meta:
//   - Purpose: Pure shape-arithmetic helper implementing CONV-1 for conv/pool layers.
//   - Usage: out := outputLen(inLen, k, s, conv.PadValid).
//   - Concurrency: Safe; pure function with no state.
//   - Related: [Conv1D], [MaxPool1D], [PadMode].
func outputLen(inLen, kernelSize, stride int, pad PadMode) int {
	if inLen <= 0 || kernelSize <= 0 || stride <= 0 {
		return 0
	}
	switch pad {
	case PadSame:
		// ceil(inLen / stride) without floating point.
		return (inLen + stride - 1) / stride
	case PadValid:
		fallthrough
	default:
		if inLen < kernelSize {
			return 0
		}
		return (inLen-kernelSize)/stride + 1
	}
}

// padSamePadding returns the (left, right) padding amounts that make Conv1D
// produce the PadSame output length. The total padding is
// `(outLen-1)*stride + kernelSize - inLen` and is split with the extra zero
// appended on the right when the total is odd.
//
// AI-Meta:
//   - Purpose: Compute the (left, right) zero-padding amounts for PadSame mode.
//   - Usage: l, r := padSamePadding(inLen, k, s).
//   - Concurrency: Safe; pure function.
//   - Related: [Conv1D.Forward], [outputLen].
func padSamePadding(inLen, kernelSize, stride int) (left, right int) {
	out := (inLen + stride - 1) / stride
	total := max((out-1)*stride+kernelSize-inLen, 0)
	left = total / 2
	right = total - left
	return
}

// Compile-time interface assertions — catch missing methods at build time (C26).
// These are populated as each concrete type is added; see the *.go siblings.

var (
	_ Layer[float32] = (*Conv1D[float32])(nil)
	_ Layer[float64] = (*Conv1D[float64])(nil)
	_ Layer[float32] = (*MaxPool1D[float32])(nil)
	_ Layer[float32] = (*AvgPool1D[float32])(nil)
	_ Layer[float32] = (*Flatten[float32])(nil)
)
