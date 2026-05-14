package conv

import (
	"fmt"

	"github.com/teratron/gonn/pkg/utils"
)

// MaxPool1D applies non-overlapping max pooling along the input feature
// vector. Stride is fixed at poolSize (non-overlapping windows, CONV-5).
// The argmax buffer is reused across iterations to route the upstream
// gradient back to the position of the max during Backward.
//
// AI-Meta:
//   - Purpose: Non-overlapping 1-D max-pool layer that records argmax positions for backward routing.
//   - Concurrency: NotSafe; mutates argmax and gradX in place.
//   - Stability: Stable.
//   - Related: [NewMaxPool1D], [Layer], [AvgPool1D].
type MaxPool1D[T utils.Float] struct {
	PoolSize int `json:"pool_size"`

	inLen  int
	argmax []int
	gradX  []T
}

// NewMaxPool1D builds a max-pool layer with the given window size. Stride
// equals poolSize (non-overlapping). Constructor panics for non-positive
// pool sizes — a programming error, not a runtime condition.
func NewMaxPool1D[T utils.Float](poolSize int) *MaxPool1D[T] {
	if poolSize <= 0 {
		panic(fmt.Sprintf("conv.NewMaxPool1D: poolSize must be positive, got %d", poolSize))
	}
	return &MaxPool1D[T]{PoolSize: poolSize}
}

// InputSize returns the input length the layer was last compiled for.
func (p *MaxPool1D[T]) InputSize() int { return p.inLen }

// OutputSize returns the number of output windows for the current input
// length. Zero when no Forward has been called yet.
func (p *MaxPool1D[T]) OutputSize() int {
	if p.inLen <= 0 {
		return 0
	}
	return outputLen(p.inLen, p.PoolSize, p.PoolSize, PadValid)
}

// Forward selects the maximum value within each non-overlapping window.
// Returns an empty slice when the pool window is larger than the input.
func (p *MaxPool1D[T]) Forward(x []T) []T {
	p.inLen = len(x)
	out := outputLen(p.inLen, p.PoolSize, p.PoolSize, PadValid)
	if out == 0 {
		p.argmax = nil
		return []T{}
	}
	if cap(p.argmax) < out {
		p.argmax = make([]int, out)
	} else {
		p.argmax = p.argmax[:out]
	}
	y := make([]T, out)
	for i := 0; i < out; i++ {
		start := i * p.PoolSize
		bestIdx := start
		best := x[start]
		for k := 1; k < p.PoolSize; k++ {
			pos := start + k
			if pos >= p.inLen {
				break
			}
			if x[pos] > best {
				best = x[pos]
				bestIdx = pos
			}
		}
		y[i] = best
		p.argmax[i] = bestIdx
	}
	return y
}

// Backward routes each upstream gradient to the position recorded in
// argmax; positions outside the argmax set receive zero (CONV-5).
func (p *MaxPool1D[T]) Backward(upstream []T) []T {
	if p.inLen == 0 || p.argmax == nil {
		return nil
	}
	if len(upstream) != len(p.argmax) {
		return nil
	}
	if cap(p.gradX) < p.inLen {
		p.gradX = make([]T, p.inLen)
	} else {
		p.gradX = p.gradX[:p.inLen]
		for i := range p.gradX {
			p.gradX[i] = 0
		}
	}
	for i, idx := range p.argmax {
		p.gradX[idx] += upstream[i]
	}
	return p.gradX
}

// GradSlots returns (nil, nil) — pooling layers have no trainable parameters.
func (p *MaxPool1D[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// Validate checks the pool size fits within the input length.
func (p *MaxPool1D[T]) Validate(inLen int) error {
	if p.PoolSize <= 0 {
		return utils.Newf(utils.ErrUserConfig, "MaxPool1D: poolSize must be positive, got %d", p.PoolSize)
	}
	if inLen > 0 && p.PoolSize > inLen {
		return fmt.Errorf("MaxPool1D: poolSize %d > input length %d: %w",
			p.PoolSize, inLen, utils.ErrConvPoolSizeMismatch)
	}
	return nil
}

// AvgPool1D applies non-overlapping average pooling. Like [MaxPool1D] but
// the backward pass distributes the upstream gradient evenly across each
// window.
//
// AI-Meta:
//   - Purpose: Non-overlapping 1-D average-pool layer; backward distributes evenly across the window.
//   - Concurrency: NotSafe; mutates gradX in place.
//   - Stability: Stable.
//   - Related: [NewAvgPool1D], [Layer], [MaxPool1D].
type AvgPool1D[T utils.Float] struct {
	PoolSize int `json:"pool_size"`

	inLen int
	gradX []T
}

// NewAvgPool1D builds an average-pool layer with the given window size.
func NewAvgPool1D[T utils.Float](poolSize int) *AvgPool1D[T] {
	if poolSize <= 0 {
		panic(fmt.Sprintf("conv.NewAvgPool1D: poolSize must be positive, got %d", poolSize))
	}
	return &AvgPool1D[T]{PoolSize: poolSize}
}

// InputSize / OutputSize / GradSlots — same shape contract as MaxPool1D.
func (p *AvgPool1D[T]) InputSize() int                { return p.inLen }
func (p *AvgPool1D[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// OutputSize returns the number of output windows.
func (p *AvgPool1D[T]) OutputSize() int {
	if p.inLen <= 0 {
		return 0
	}
	return outputLen(p.inLen, p.PoolSize, p.PoolSize, PadValid)
}

// Forward averages each non-overlapping window. Windows that overrun the
// input boundary contribute only the in-range elements (no zero-padding).
func (p *AvgPool1D[T]) Forward(x []T) []T {
	p.inLen = len(x)
	out := outputLen(p.inLen, p.PoolSize, p.PoolSize, PadValid)
	if out == 0 {
		return []T{}
	}
	y := make([]T, out)
	for i := 0; i < out; i++ {
		start := i * p.PoolSize
		var sum T
		count := 0
		for k := 0; k < p.PoolSize; k++ {
			pos := start + k
			if pos >= p.inLen {
				break
			}
			sum += x[pos]
			count++
		}
		if count > 0 {
			y[i] = sum / T(count)
		}
	}
	return y
}

// Backward distributes each upstream gradient evenly across the window
// positions (1 / poolSize per position).
func (p *AvgPool1D[T]) Backward(upstream []T) []T {
	if p.inLen == 0 || len(upstream) == 0 {
		return nil
	}
	if cap(p.gradX) < p.inLen {
		p.gradX = make([]T, p.inLen)
	} else {
		p.gradX = p.gradX[:p.inLen]
		for i := range p.gradX {
			p.gradX[i] = 0
		}
	}
	inv := T(1) / T(p.PoolSize)
	for i, up := range upstream {
		start := i * p.PoolSize
		share := up * inv
		for k := 0; k < p.PoolSize; k++ {
			pos := start + k
			if pos >= p.inLen {
				break
			}
			p.gradX[pos] += share
		}
	}
	return p.gradX
}

// Validate mirrors the MaxPool1D rule set.
func (p *AvgPool1D[T]) Validate(inLen int) error {
	if p.PoolSize <= 0 {
		return utils.Newf(utils.ErrUserConfig, "AvgPool1D: poolSize must be positive, got %d", p.PoolSize)
	}
	if inLen > 0 && p.PoolSize > inLen {
		return fmt.Errorf("AvgPool1D: poolSize %d > input length %d: %w",
			p.PoolSize, inLen, utils.ErrConvPoolSizeMismatch)
	}
	return nil
}
