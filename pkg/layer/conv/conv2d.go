package conv

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/utils"
)

// Conv2D is a 2-D convolutional layer with NumFilters kernels of spatial
// extent (KernelH × KernelW) sliding over a CHW-laid-out input feature map.
//
// Tensor layout follows CONV2D-C9: input element (c, y, x) resides at
// flat index c*H*W + y*W + x; filter element (f, c, i, j) resides at
// f*(InChannels*KernelH*KernelW) + c*(KernelH*KernelW) + i*KernelW + j.
// This generalises Conv1D Variant A storage by adding the inner-channel
// axis and matches PyTorch / cuDNN convention.
//
// AI-Meta:
//   - Purpose: 2-D convolutional layer producing NumFilters output channels by sliding-window cross-correlation over CHW input.
//   - Concurrency: NotSafe; Forward and Backward mutate per-iteration buffers (lastInput, gradW, gradB, gradX).
//   - Stability: Stable.
//   - Related: [NewConv2D], [Layer], [MaxPool2D], [AvgPool2D], [Flatten2D].
type Conv2D[T utils.Float] struct {
	gradB      []T     `json:"-"`
	Weights    []T     `json:"weights"`
	Biases     []T     `json:"biases,omitempty"`
	gradW      []T     `json:"-"`
	gradX      []T     `json:"-"`
	lastInput  []T     `json:"-"`
	InH        int     `json:"in_h"`
	InW        int     `json:"in_w"`
	NumFilters int     `json:"num_filters"`
	InChannels int     `json:"in_channels"`
	KernelH    int     `json:"kernel_h"`
	KernelW    int     `json:"kernel_w"`
	StrideH    int     `json:"stride_h"`
	StrideW    int     `json:"stride_w"`
	Padding    PadMode `json:"padding"`
	UseBias    bool    `json:"use_bias"`
}

// NewConv2D builds a Conv2D layer. The weight buffer is allocated but left
// at the type's zero value — call [Conv2D.Init] (or compile() via the
// network builder) to populate weights via [utils.HeNormal].
//
// AI-Meta:
//   - Purpose: Construct a Conv2D layer with weight buffer sized NumFilters*InChannels*KernelH*KernelW.
//   - Usage: c := conv.NewConv2D[float32](8, 1, 3, 3, 1, 1, conv.PadValid, true).
//   - Related: [Conv2D], [Conv2D.Init], [WithConv2D].
//   - Stability: Stable.
func NewConv2D[T utils.Float](numFilters, inChannels, kernelH, kernelW, strideH, strideW int, pad PadMode, useBias bool) *Conv2D[T] {
	if numFilters <= 0 {
		numFilters = 1
	}
	if inChannels <= 0 {
		inChannels = 1
	}
	if kernelH <= 0 {
		kernelH = 1
	}
	if kernelW <= 0 {
		kernelW = 1
	}
	if strideH <= 0 {
		strideH = 1
	}
	if strideW <= 0 {
		strideW = 1
	}
	c := &Conv2D[T]{
		NumFilters: numFilters,
		InChannels: inChannels,
		KernelH:    kernelH,
		KernelW:    kernelW,
		StrideH:    strideH,
		StrideW:    strideW,
		Padding:    pad,
		UseBias:    useBias,
		Weights:    make([]T, numFilters*inChannels*kernelH*kernelW),
	}
	if useBias {
		c.Biases = make([]T, numFilters)
	}
	return c
}

// Init populates the weight buffer using He-normal sampling (CONV2D-8). The
// fan-in for each filter is InChannels*KernelH*KernelW (the per-output
// activation receptive volume); biases are zero-initialised.
//
// AI-Meta:
//   - Purpose: Sample weights via He-normal distribution; zero-initialise biases.
//   - Usage: c.Init(rng).
//   - Concurrency: NotSafe; uses rng, must not be shared.
//   - Related: [utils.HeNormal], [NewConv2D].
//   - Stability: Stable.
func (c *Conv2D[T]) Init(rng *rand.Rand) {
	if rng == nil {
		panic("conv.Conv2D.Init: nil *rand.Rand")
	}
	fanIn := c.InChannels * c.KernelH * c.KernelW
	for i := range c.Weights {
		c.Weights[i] = utils.HeNormal[T](rng, fanIn)
	}
	if c.UseBias {
		for i := range c.Biases {
			c.Biases[i] = 0
		}
	}
}

// SetInputShape records the spatial dimensions for downstream OutputSize
// queries during compile(). Called once at compile time before the first
// Forward; InH and InW are otherwise inferred from the first Forward call
// when InChannels == 1 and len(x) is a perfect square.
func (c *Conv2D[T]) SetInputShape(inH, inW int) {
	c.InH = inH
	c.InW = inW
}

// InputSize returns the flat input length (InChannels*InH*InW). Zero before
// SetInputShape or the first Forward call.
func (c *Conv2D[T]) InputSize() int {
	if c.InH <= 0 || c.InW <= 0 {
		return 0
	}
	return c.InChannels * c.InH * c.InW
}

// OutputSize returns the flat output length (NumFilters*outH*outW). Zero
// before SetInputShape or the first Forward call.
func (c *Conv2D[T]) OutputSize() int {
	if c.InH <= 0 || c.InW <= 0 {
		return 0
	}
	outH, outW := outputShape(c.InH, c.InW, c.KernelH, c.KernelW, c.StrideH, c.StrideW, c.Padding)
	return c.NumFilters * outH * outW
}

// OutputShape returns (outH, outW) for the layer's currently configured
// input dimensions. Returns (0, 0) before SetInputShape or the first
// Forward call.
func (c *Conv2D[T]) OutputShape() (outH, outW int) {
	if c.InH <= 0 || c.InW <= 0 {
		return 0, 0
	}
	return outputShape(c.InH, c.InW, c.KernelH, c.KernelW, c.StrideH, c.StrideW, c.Padding)
}

// Forward applies the 2-D convolutional cross-correlation to x. Input MUST
// be a CHW-flat slice of length InChannels*InH*InW; output is filter-major
// CHW-flat of length NumFilters*outH*outW. Returns an empty slice when the
// input spatial extent is smaller than the kernel under PadValid.
//
// AI-Meta:
//   - Purpose: Compute sliding-window cross-correlation for every filter over CHW input; output is filter-major CHW.
//   - Usage: y := c.Forward(x).
//   - Concurrency: NotSafe; mutates lastInput.
//   - Related: [Conv2D.Backward], [outputShape].
//   - Stability: Stable.
func (c *Conv2D[T]) Forward(x []T) []T {
	if c.InH <= 0 || c.InW <= 0 {
		// Spatial dims not yet set — infer assuming square input only when
		// InChannels == 1 and len(x) is a perfect square; otherwise reject.
		if c.InChannels == 1 {
			n := len(x)
			s := isqrt(n)
			if s*s == n {
				c.InH, c.InW = s, s
			} else {
				return []T{}
			}
		} else {
			return []T{}
		}
	}
	if len(x) != c.InChannels*c.InH*c.InW {
		return []T{}
	}
	outH, outW := outputShape(c.InH, c.InW, c.KernelH, c.KernelW, c.StrideH, c.StrideW, c.Padding)
	if outH == 0 || outW == 0 {
		c.lastInput = nil
		return []T{}
	}
	c.lastInput = append(c.lastInput[:0], x...)
	out := make([]T, c.NumFilters*outH*outW)

	var padTop, padLeft int
	if c.Padding == PadSame {
		padTop, _ = padSamePadding(c.InH, c.KernelH, c.StrideH)
		padLeft, _ = padSamePadding(c.InW, c.KernelW, c.StrideW)
	}

	kHW := c.KernelH * c.KernelW
	cKHW := c.InChannels * kHW
	hw := c.InH * c.InW
	oHW := outH * outW

	for f := range c.NumFilters {
		wfBase := f * cKHW
		ofBase := f * oHW
		for oy := range outH {
			for ox := range outW {
				var acc T
				iyBase := oy*c.StrideH - padTop
				ixBase := ox*c.StrideW - padLeft
				for ch := range c.InChannels {
					wcBase := wfBase + ch*kHW
					xcBase := ch * hw
					for ky := range c.KernelH {
						iy := iyBase + ky
						if iy < 0 || iy >= c.InH {
							continue
						}
						wkyBase := wcBase + ky*c.KernelW
						xyBase := xcBase + iy*c.InW
						for kx := range c.KernelW {
							ix := ixBase + kx
							if ix < 0 || ix >= c.InW {
								continue
							}
							acc += x[xyBase+ix] * c.Weights[wkyBase+kx]
						}
					}
				}
				if c.UseBias {
					acc += c.Biases[f]
				}
				out[ofBase+oy*outW+ox] = acc
			}
		}
	}
	return out
}

// Backward computes ∂L/∂X given ∂L/∂Y and accumulates ∂L/∂W and ∂L/∂B in
// the internal grad buffers reachable via [Conv2D.GradSlots]. The grad
// buffers are zero-reset at the start of each call; callers MUST consume
// them via the optimizer before the next Backward.
//
// Per CONV2D-4: gradW[f,c,i,j] += sum_{oy,ox} lastInput[c, oy*sH+i-padTop,
// ox*sW+j-padLeft] * upstream[f,oy,ox]; gradX[c,iy,ix] += sum_{f,ky,kx}
// Weights[f,c,ky,kx] * upstream[f, (iy+padTop-ky)/sH, (ix+padLeft-kx)/sW]
// (only when the division is exact and the result lies in the output range).
// Bias gradient gradB[f] = sum_{oy,ox} upstream[f,oy,ox].
//
// AI-Meta:
//   - Purpose: Compute input-side gradient and accumulate weight/bias gradients per CONV2D-4.
//   - Usage: gradX := c.Backward(upstream).
//   - Concurrency: NotSafe; mutates gradW, gradB, gradX; lastInput must be the most recent Forward result.
//   - Related: [Conv2D.Forward], [Conv2D.GradSlots].
//   - Stability: Stable.
func (c *Conv2D[T]) Backward(upstream []T) []T {
	if c.InH <= 0 || c.InW <= 0 || c.lastInput == nil {
		return nil
	}
	outH, outW := outputShape(c.InH, c.InW, c.KernelH, c.KernelW, c.StrideH, c.StrideW, c.Padding)
	if outH == 0 || outW == 0 {
		return nil
	}
	if len(upstream) != c.NumFilters*outH*outW {
		return nil
	}

	// Cap-check + zero-reset gradient buffers (PERF-4 zero-alloc steady-state).
	wLen := len(c.Weights)
	if cap(c.gradW) < wLen {
		c.gradW = make([]T, wLen)
	} else {
		c.gradW = c.gradW[:wLen]
		for i := range c.gradW {
			c.gradW[i] = 0
		}
	}
	if c.UseBias {
		if cap(c.gradB) < c.NumFilters {
			c.gradB = make([]T, c.NumFilters)
		} else {
			c.gradB = c.gradB[:c.NumFilters]
			for i := range c.gradB {
				c.gradB[i] = 0
			}
		}
	} else {
		c.gradB = nil
	}
	xLen := c.InChannels * c.InH * c.InW
	if cap(c.gradX) < xLen {
		c.gradX = make([]T, xLen)
	} else {
		c.gradX = c.gradX[:xLen]
		for i := range c.gradX {
			c.gradX[i] = 0
		}
	}

	var padTop, padLeft int
	if c.Padding == PadSame {
		padTop, _ = padSamePadding(c.InH, c.KernelH, c.StrideH)
		padLeft, _ = padSamePadding(c.InW, c.KernelW, c.StrideW)
	}

	kHW := c.KernelH * c.KernelW
	cKHW := c.InChannels * kHW
	hw := c.InH * c.InW
	oHW := outH * outW

	for f := range c.NumFilters {
		wfBase := f * cKHW
		ofBase := f * oHW
		for oy := range outH {
			for ox := range outW {
				up := upstream[ofBase+oy*outW+ox]
				if c.UseBias {
					c.gradB[f] += up
				}
				iyBase := oy*c.StrideH - padTop
				ixBase := ox*c.StrideW - padLeft
				for ch := range c.InChannels {
					wcBase := wfBase + ch*kHW
					xcBase := ch * hw
					for ky := range c.KernelH {
						iy := iyBase + ky
						if iy < 0 || iy >= c.InH {
							continue
						}
						wkyBase := wcBase + ky*c.KernelW
						xyBase := xcBase + iy*c.InW
						for kx := range c.KernelW {
							ix := ixBase + kx
							if ix < 0 || ix >= c.InW {
								continue
							}
							c.gradW[wkyBase+kx] += c.lastInput[xyBase+ix] * up
							c.gradX[xyBase+ix] += c.Weights[wkyBase+kx] * up
						}
					}
				}
			}
		}
	}
	return c.gradX
}

// GradSlots returns the gradient accumulators for the optimizer Step()
// call. Both slices share storage with the layer; the caller MUST NOT
// retain references past the next Forward / Backward pair.
func (c *Conv2D[T]) GradSlots() (gradW, gradB []T) {
	return c.gradW, c.gradB
}

// Validate verifies the layer is well-formed at compile time given a
// concrete (inC, inH, inW) shape. Returns an error wrapping
// [utils.ErrUserConfig] for parameter mistakes and
// [utils.ErrConv2DShapeMismatch] for shape violations under PadValid
// (CONV2D-1).
//
// AI-Meta:
//   - Purpose: Validate Conv2D parameters at network compile time.
//   - Usage: if err := c.Validate(inC, inH, inW); err != nil { return err }.
//   - Concurrency: Safe; read-only access to layer fields.
//   - Related: [Conv2D], [outputShape].
//   - Stability: Stable.
func (c *Conv2D[T]) Validate(inC, inH, inW int) error {
	if c.NumFilters <= 0 {
		return utils.Newf(utils.ErrUserConfig, "Conv2D: numFilters must be positive, got %d", c.NumFilters)
	}
	if c.InChannels <= 0 {
		return utils.Newf(utils.ErrUserConfig, "Conv2D: inChannels must be positive, got %d", c.InChannels)
	}
	if inC > 0 && inC != c.InChannels {
		return utils.Newf(utils.ErrUserConfig, "Conv2D: inChannels mismatch — expected %d, got %d", c.InChannels, inC)
	}
	if c.KernelH <= 0 || c.KernelW <= 0 {
		return utils.Newf(utils.ErrUserConfig, "Conv2D: kernel dimensions must be positive, got (%d, %d)", c.KernelH, c.KernelW)
	}
	if c.StrideH <= 0 || c.StrideW <= 0 {
		return utils.Newf(utils.ErrUserConfig, "Conv2D: stride must be positive, got (%d, %d)", c.StrideH, c.StrideW)
	}
	if c.Padding == PadValid {
		if inH > 0 && inH < c.KernelH {
			return fmt.Errorf("Conv2D: input H %d < kernel H %d under PadValid: %w",
				inH, c.KernelH, utils.ErrConv2DShapeMismatch)
		}
		if inW > 0 && inW < c.KernelW {
			return fmt.Errorf("Conv2D: input W %d < kernel W %d under PadValid: %w",
				inW, c.KernelW, utils.ErrConv2DShapeMismatch)
		}
	}
	expectedW := c.NumFilters * c.InChannels * c.KernelH * c.KernelW
	if len(c.Weights) != expectedW {
		return utils.NewIntegrityError("Conv2D.weights",
			fmt.Sprintf("%d", expectedW),
			fmt.Sprintf("%d", len(c.Weights)))
	}
	if c.UseBias && len(c.Biases) != c.NumFilters {
		return utils.NewIntegrityError("Conv2D.biases",
			fmt.Sprintf("%d", c.NumFilters),
			fmt.Sprintf("%d", len(c.Biases)))
	}
	return nil
}

// MarshalJSON implements [json.Marshaler] for round-trip persistence
// (CONV2D-7). Non-exported scratch buffers are excluded via the struct-alias
// trick (alias drops MarshalJSON to avoid recursion).
func (c *Conv2D[T]) MarshalJSON() ([]byte, error) {
	type alias Conv2D[T]
	return json.Marshal((*alias)(c))
}

// UnmarshalJSON implements [json.Unmarshaler] for round-trip persistence
// (CONV2D-7). Verifies the weight buffer length matches the declared shape.
func (c *Conv2D[T]) UnmarshalJSON(data []byte) error {
	type alias Conv2D[T]
	if err := json.Unmarshal(data, (*alias)(c)); err != nil {
		return utils.Wrap(utils.ErrIntegrity, err, "Conv2D.UnmarshalJSON")
	}
	if c.NumFilters <= 0 || c.InChannels <= 0 || c.KernelH <= 0 || c.KernelW <= 0 || c.StrideH <= 0 || c.StrideW <= 0 {
		return utils.Newf(utils.ErrIntegrity,
			"Conv2D.UnmarshalJSON: invalid shape numFilters=%d inChannels=%d kernel=(%d,%d) stride=(%d,%d)",
			c.NumFilters, c.InChannels, c.KernelH, c.KernelW, c.StrideH, c.StrideW)
	}
	expectedW := c.NumFilters * c.InChannels * c.KernelH * c.KernelW
	if len(c.Weights) != expectedW {
		return utils.NewIntegrityError("Conv2D.weights",
			fmt.Sprintf("%d", expectedW),
			fmt.Sprintf("%d", len(c.Weights)))
	}
	if c.UseBias && len(c.Biases) != c.NumFilters {
		return utils.NewIntegrityError("Conv2D.biases",
			fmt.Sprintf("%d", c.NumFilters),
			fmt.Sprintf("%d", len(c.Biases)))
	}
	return nil
}

// outputShape returns (outH, outW) for a 2-D convolution given input
// dimensions and kernel parameters. Generalises [outputLen] to two axes
// per CONV2D-1.
//
// AI-Meta:
//   - Purpose: Pure shape-arithmetic helper for Conv2D / Pool2D output dimensions.
//   - Usage: outH, outW := outputShape(28, 28, 3, 3, 1, 1, conv.PadValid).
//   - Concurrency: Safe; pure function.
//   - Related: [outputLen], [Conv2D], [MaxPool2D].
func outputShape(inH, inW, kH, kW, sH, sW int, pad PadMode) (outH, outW int) {
	outH = outputLen(inH, kH, sH, pad)
	outW = outputLen(inW, kW, sW, pad)
	return
}

// isqrt is a small integer square-root helper used by [Conv2D.Forward] when
// it must infer a square spatial shape from a single-channel flat input
// (e.g. MNIST 784-byte tensor before WithImageShape is applied).
func isqrt(n int) int {
	if n <= 0 {
		return 0
	}
	x := n
	y := (x + 1) / 2
	for y < x {
		x = y
		y = (x + n/x) / 2
	}
	return x
}
