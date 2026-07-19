package conv

import (
	"fmt"

	"github.com/teratron/gonn/pkg/utils"
)

// MaxPool2D applies non-overlapping 2-D max pooling per channel. Stride
// equals (PoolH, PoolW) — non-overlapping windows (CONV2D-5). Argmax
// (argmaxH, argmaxW) buffers are reused across iterations to route the
// upstream gradient back to the position of the max during Backward.
//
// AI-Meta:
//   - Purpose: Non-overlapping 2-D max-pool layer that records argmax positions for backward routing.
//   - Concurrency: NotSafe; mutates argmaxH/argmaxW and gradX in place.
//   - Stability: Stable.
//   - Related: [NewMaxPool2D], [Layer], [AvgPool2D].
type MaxPool2D[T utils.Float] struct {
	argmaxH    []int
	argmaxW    []int
	gradX      []T
	PoolH      int `json:"pool_h"`
	PoolW      int `json:"pool_w"`
	InChannels int `json:"in_channels,omitempty"`
	InH        int `json:"in_h,omitempty"`
	InW        int `json:"in_w,omitempty"`
}

// NewMaxPool2D builds a 2-D max-pool layer with the given window
// dimensions. Stride equals (poolH, poolW). Constructor panics for
// non-positive dimensions — a programming error, not a runtime condition.
func NewMaxPool2D[T utils.Float](poolH, poolW int) *MaxPool2D[T] {
	if poolH <= 0 || poolW <= 0 {
		panic(fmt.Sprintf("conv.NewMaxPool2D: pool dims must be positive, got (%d, %d)", poolH, poolW))
	}
	return &MaxPool2D[T]{PoolH: poolH, PoolW: poolW}
}

// SetInputShape records the spatial dimensions for OutputSize queries.
func (p *MaxPool2D[T]) SetInputShape(inC, inH, inW int) {
	p.InChannels = inC
	p.InH = inH
	p.InW = inW
}

// InputSize returns the flat input length (InChannels*InH*InW).
func (p *MaxPool2D[T]) InputSize() int {
	if p.InH <= 0 || p.InW <= 0 || p.InChannels <= 0 {
		return 0
	}
	return p.InChannels * p.InH * p.InW
}

// OutputSize returns the flat output length (InChannels*outH*outW).
func (p *MaxPool2D[T]) OutputSize() int {
	if p.InH <= 0 || p.InW <= 0 || p.InChannels <= 0 {
		return 0
	}
	outH, outW := p.outputShape()
	return p.InChannels * outH * outW
}

// OutputShape returns (outH, outW) for the layer's configured input.
func (p *MaxPool2D[T]) OutputShape() (outH, outW int) {
	if p.InH <= 0 || p.InW <= 0 {
		return 0, 0
	}
	return p.outputShape()
}

func (p *MaxPool2D[T]) outputShape() (outH, outW int) {
	outH = outputLen(p.InH, p.PoolH, p.PoolH, PadValid)
	outW = outputLen(p.InW, p.PoolW, p.PoolW, PadValid)
	return
}

// Forward selects the maximum value within each non-overlapping window
// per channel independently. Input is CHW-flat of length
// InChannels*InH*InW; output is CHW-flat of length InChannels*outH*outW.
func (p *MaxPool2D[T]) Forward(x []T) []T {
	if p.InChannels <= 0 || p.InH <= 0 || p.InW <= 0 {
		// Infer single-channel square shape when feasible.
		s := isqrt(len(x))
		if s*s == len(x) {
			p.InChannels, p.InH, p.InW = 1, s, s
		} else {
			return []T{}
		}
	}
	if len(x) != p.InChannels*p.InH*p.InW {
		return []T{}
	}
	outH, outW := p.outputShape()
	if outH == 0 || outW == 0 {
		p.argmaxH = nil
		p.argmaxW = nil
		return []T{}
	}
	total := p.InChannels * outH * outW
	if cap(p.argmaxH) < total {
		p.argmaxH = make([]int, total)
	} else {
		p.argmaxH = p.argmaxH[:total]
	}
	if cap(p.argmaxW) < total {
		p.argmaxW = make([]int, total)
	} else {
		p.argmaxW = p.argmaxW[:total]
	}
	out := make([]T, total)

	hw := p.InH * p.InW
	oHW := outH * outW

	for ch := range p.InChannels {
		xcBase := ch * hw
		ocBase := ch * oHW
		for oy := range outH {
			for ox := range outW {
				idx := ocBase + oy*outW + ox
				iyStart := oy * p.PoolH
				ixStart := ox * p.PoolW
				best := x[xcBase+iyStart*p.InW+ixStart]
				bestY, bestX := iyStart, ixStart
				for ky := range p.PoolH {
					iy := iyStart + ky
					if iy >= p.InH {
						break
					}
					xyBase := xcBase + iy*p.InW
					for kx := range p.PoolW {
						ix := ixStart + kx
						if ix >= p.InW {
							break
						}
						v := x[xyBase+ix]
						if v > best {
							best = v
							bestY = iy
							bestX = ix
						}
					}
				}
				out[idx] = best
				p.argmaxH[idx] = bestY
				p.argmaxW[idx] = bestX
			}
		}
	}
	return out
}

// Backward routes each upstream gradient to the position recorded in
// (argmaxH, argmaxW); positions outside the argmax set receive zero
// (CONV2D-5).
func (p *MaxPool2D[T]) Backward(upstream []T) []T {
	if p.InChannels <= 0 || p.InH <= 0 || p.InW <= 0 || p.argmaxH == nil {
		return nil
	}
	outH, outW := p.outputShape()
	if outH == 0 || outW == 0 {
		return nil
	}
	if len(upstream) != p.InChannels*outH*outW {
		return nil
	}
	xLen := p.InChannels * p.InH * p.InW
	if cap(p.gradX) < xLen {
		p.gradX = make([]T, xLen)
	} else {
		p.gradX = p.gradX[:xLen]
		for i := range p.gradX {
			p.gradX[i] = 0
		}
	}
	hw := p.InH * p.InW
	oHW := outH * outW
	for ch := range p.InChannels {
		xcBase := ch * hw
		ocBase := ch * oHW
		for oy := range outH {
			for ox := range outW {
				idx := ocBase + oy*outW + ox
				iy := p.argmaxH[idx]
				ix := p.argmaxW[idx]
				p.gradX[xcBase+iy*p.InW+ix] += upstream[idx]
			}
		}
	}
	return p.gradX
}

// GradSlots returns (nil, nil) — pooling layers have no trainable parameters.
func (p *MaxPool2D[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// Validate checks the pool dims fit within the input.
func (p *MaxPool2D[T]) Validate(inC, inH, inW int) error {
	if p.PoolH <= 0 || p.PoolW <= 0 {
		return utils.Newf(utils.ErrUserConfig, "MaxPool2D: pool dims must be positive, got (%d, %d)", p.PoolH, p.PoolW)
	}
	if inH > 0 && p.PoolH > inH {
		return fmt.Errorf("MaxPool2D: poolH %d > input H %d: %w",
			p.PoolH, inH, utils.ErrConvPoolSizeMismatch)
	}
	if inW > 0 && p.PoolW > inW {
		return fmt.Errorf("MaxPool2D: poolW %d > input W %d: %w",
			p.PoolW, inW, utils.ErrConvPoolSizeMismatch)
	}
	_ = inC
	return nil
}

// AvgPool2D applies non-overlapping 2-D average pooling per channel.
// Like [MaxPool2D] but the backward pass distributes the upstream gradient
// evenly across each window (CONV2D-5).
//
// AI-Meta:
//   - Purpose: Non-overlapping 2-D average-pool layer; backward distributes evenly across the window.
//   - Concurrency: NotSafe; mutates gradX in place.
//   - Stability: Stable.
//   - Related: [NewAvgPool2D], [Layer], [MaxPool2D].
type AvgPool2D[T utils.Float] struct {
	gradX      []T
	PoolH      int `json:"pool_h"`
	PoolW      int `json:"pool_w"`
	InChannels int `json:"in_channels,omitempty"`
	InH        int `json:"in_h,omitempty"`
	InW        int `json:"in_w,omitempty"`
}

// NewAvgPool2D builds a 2-D average-pool layer.
func NewAvgPool2D[T utils.Float](poolH, poolW int) *AvgPool2D[T] {
	if poolH <= 0 || poolW <= 0 {
		panic(fmt.Sprintf("conv.NewAvgPool2D: pool dims must be positive, got (%d, %d)", poolH, poolW))
	}
	return &AvgPool2D[T]{PoolH: poolH, PoolW: poolW}
}

// SetInputShape records the spatial dimensions for OutputSize queries.
func (p *AvgPool2D[T]) SetInputShape(inC, inH, inW int) {
	p.InChannels = inC
	p.InH = inH
	p.InW = inW
}

// InputSize / OutputSize / GradSlots — same shape contract as MaxPool2D.
func (p *AvgPool2D[T]) InputSize() int {
	if p.InH <= 0 || p.InW <= 0 || p.InChannels <= 0 {
		return 0
	}
	return p.InChannels * p.InH * p.InW
}

func (p *AvgPool2D[T]) OutputSize() int {
	if p.InH <= 0 || p.InW <= 0 || p.InChannels <= 0 {
		return 0
	}
	outH, outW := p.outputShape()
	return p.InChannels * outH * outW
}

func (p *AvgPool2D[T]) OutputShape() (outH, outW int) {
	if p.InH <= 0 || p.InW <= 0 {
		return 0, 0
	}
	return p.outputShape()
}

func (p *AvgPool2D[T]) outputShape() (outH, outW int) {
	outH = outputLen(p.InH, p.PoolH, p.PoolH, PadValid)
	outW = outputLen(p.InW, p.PoolW, p.PoolW, PadValid)
	return
}

// GradSlots returns (nil, nil) — pooling layers have no trainable parameters.
func (p *AvgPool2D[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// Forward averages each non-overlapping window per channel. Windows that
// overrun the input boundary contribute only the in-range elements
// (no zero-padding).
func (p *AvgPool2D[T]) Forward(x []T) []T {
	if p.InChannels <= 0 || p.InH <= 0 || p.InW <= 0 {
		s := isqrt(len(x))
		if s*s == len(x) {
			p.InChannels, p.InH, p.InW = 1, s, s
		} else {
			return []T{}
		}
	}
	if len(x) != p.InChannels*p.InH*p.InW {
		return []T{}
	}
	outH, outW := p.outputShape()
	if outH == 0 || outW == 0 {
		return []T{}
	}
	total := p.InChannels * outH * outW
	out := make([]T, total)
	hw := p.InH * p.InW
	oHW := outH * outW

	for ch := range p.InChannels {
		xcBase := ch * hw
		ocBase := ch * oHW
		for oy := range outH {
			for ox := range outW {
				iyStart := oy * p.PoolH
				ixStart := ox * p.PoolW
				var sum T
				count := 0
				for ky := range p.PoolH {
					iy := iyStart + ky
					if iy >= p.InH {
						break
					}
					xyBase := xcBase + iy*p.InW
					for kx := range p.PoolW {
						ix := ixStart + kx
						if ix >= p.InW {
							break
						}
						sum += x[xyBase+ix]
						count++
					}
				}
				if count > 0 {
					out[ocBase+oy*outW+ox] = sum / T(count)
				}
			}
		}
	}
	return out
}

// Backward distributes each upstream gradient evenly across the window
// positions (1 / (PoolH*PoolW) per position).
func (p *AvgPool2D[T]) Backward(upstream []T) []T {
	if p.InChannels <= 0 || p.InH <= 0 || p.InW <= 0 {
		return nil
	}
	outH, outW := p.outputShape()
	if outH == 0 || outW == 0 {
		return nil
	}
	if len(upstream) != p.InChannels*outH*outW {
		return nil
	}
	xLen := p.InChannels * p.InH * p.InW
	if cap(p.gradX) < xLen {
		p.gradX = make([]T, xLen)
	} else {
		p.gradX = p.gradX[:xLen]
		for i := range p.gradX {
			p.gradX[i] = 0
		}
	}
	inv := T(1) / T(p.PoolH*p.PoolW)
	hw := p.InH * p.InW
	oHW := outH * outW
	for ch := range p.InChannels {
		xcBase := ch * hw
		ocBase := ch * oHW
		for oy := range outH {
			for ox := range outW {
				share := upstream[ocBase+oy*outW+ox] * inv
				iyStart := oy * p.PoolH
				ixStart := ox * p.PoolW
				for ky := range p.PoolH {
					iy := iyStart + ky
					if iy >= p.InH {
						break
					}
					xyBase := xcBase + iy*p.InW
					for kx := range p.PoolW {
						ix := ixStart + kx
						if ix >= p.InW {
							break
						}
						p.gradX[xyBase+ix] += share
					}
				}
			}
		}
	}
	return p.gradX
}

// Validate mirrors the MaxPool2D rule set.
func (p *AvgPool2D[T]) Validate(inC, inH, inW int) error {
	if p.PoolH <= 0 || p.PoolW <= 0 {
		return utils.Newf(utils.ErrUserConfig, "AvgPool2D: pool dims must be positive, got (%d, %d)", p.PoolH, p.PoolW)
	}
	if inH > 0 && p.PoolH > inH {
		return fmt.Errorf("AvgPool2D: poolH %d > input H %d: %w",
			p.PoolH, inH, utils.ErrConvPoolSizeMismatch)
	}
	if inW > 0 && p.PoolW > inW {
		return fmt.Errorf("AvgPool2D: poolW %d > input W %d: %w",
			p.PoolW, inW, utils.ErrConvPoolSizeMismatch)
	}
	_ = inC
	return nil
}
