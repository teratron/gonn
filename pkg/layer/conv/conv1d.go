package conv

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/utils"
)

// Conv1D is a 1-D convolutional layer with numFilters kernels of length
// kernelSize sliding over the input with the given stride and padding.
//
// Weight storage uses Variant A — a single flattened slice of length
// numFilters*kernelSize laid out filter-major (filter 0 occupies
// weights[0:kernelSize], filter 1 occupies weights[kernelSize:2*kernelSize],
// and so on). This choice keeps a single allocation, gives the inner
// cross-correlation loop predictable cache behaviour, and lets the gradient
// scratch buffer reuse the same shape for sync.Pool reuse (PERF-4).
//
// AI-Meta:
//   - Purpose: 1-D convolutional layer producing numFilters output channels by sliding-window cross-correlation.
//   - Concurrency: NotSafe; Forward and Backward mutate per-iteration buffers (lastInput, lastOutput, gradW, gradB).
//   - Stability: Stable.
//   - Related: [NewConv1D], [Layer], [MaxPool1D], [Flatten].
type Conv1D[T utils.Float] struct {
	gradB      []T     `json:"-"`
	Weights    []T     `json:"weights"`
	Biases     []T     `json:"biases,omitempty"`
	gradW      []T     `json:"-"`
	gradX      []T     `json:"-"`
	lastInput  []T     `json:"-"`
	lastOutput []T     `json:"-"`
	KernelSize int     `json:"kernel_size"`
	Stride     int     `json:"stride"`
	InLen      int     `json:"in_len"`
	NumFilters int     `json:"num_filters"`
	Padding    PadMode `json:"padding"`
	UseBias    bool    `json:"use_bias"`
}

// NewConv1D builds a Conv1D layer. The weight buffer is allocated but left
// at the type's zero value — call [Conv1D.Init] (or compile() via the
// network builder) to populate weights via [utils.HeNormal] / similar.
//
// AI-Meta:
//   - Purpose: Construct a Conv1D layer with weight buffer sized numFilters*kernelSize.
//   - Usage: c := conv.NewConv1D[float32](16, 3, 1, conv.PadValid, true).
//   - Related: [Conv1D], [Conv1D.Init], [WithConv1D].
//   - Stability: Stable.
func NewConv1D[T utils.Float](numFilters, kernelSize, stride int, pad PadMode, useBias bool) *Conv1D[T] {
	if numFilters <= 0 {
		numFilters = 1
	}
	if kernelSize <= 0 {
		kernelSize = 1
	}
	if stride <= 0 {
		stride = 1
	}
	c := &Conv1D[T]{
		NumFilters: numFilters,
		KernelSize: kernelSize,
		Stride:     stride,
		Padding:    pad,
		UseBias:    useBias,
		Weights:    make([]T, numFilters*kernelSize),
	}
	if useBias {
		c.Biases = make([]T, numFilters)
	}
	return c
}

// Init populates the weight buffer using He-normal sampling (CONV-8). The
// fan-in for each filter is kernelSize; biases are zero-initialised.
//
// AI-Meta:
//   - Purpose: Sample weights via He-normal distribution; zero-initialise biases.
//   - Usage: c.Init(rng).
//   - Concurrency: NotSafe; uses rng, must not be shared.
//   - Related: [utils.HeNormal], [NewConv1D].
//   - Stability: Stable.
func (c *Conv1D[T]) Init(rng *rand.Rand) {
	if rng == nil {
		panic("conv.Conv1D.Init: nil *rand.Rand")
	}
	for i := range c.Weights {
		c.Weights[i] = utils.HeNormal[T](rng, c.KernelSize)
	}
	if c.UseBias {
		for i := range c.Biases {
			c.Biases[i] = 0
		}
	}
}

// InputSize returns the input length the layer was last compiled or read
// from JSON for. Zero before the first Forward call on a fresh layer.
func (c *Conv1D[T]) InputSize() int { return c.InLen }

// OutputSize returns the total flat length of the output feature map:
// outputLen(InLen) * NumFilters. Returns 0 before the first Forward call.
func (c *Conv1D[T]) OutputSize() int {
	if c.InLen <= 0 {
		return 0
	}
	per := outputLen(c.InLen, c.KernelSize, c.Stride, c.Padding)
	return per * c.NumFilters
}

// Forward applies the convolutional cross-correlation to x. The output is
// laid out filter-major: filter 0 spans output[0:outPerFilter], filter 1
// spans output[outPerFilter:2*outPerFilter], and so on. Returns an empty
// slice when the input is shorter than the kernel under PadValid.
//
// AI-Meta:
//   - Purpose: Compute sliding-window cross-correlation for every filter; output is filter-major.
//   - Usage: y := c.Forward(x).
//   - Concurrency: NotSafe; mutates lastInput / lastOutput.
//   - Related: [Conv1D.Backward], [outputLen].
//   - Stability: Stable.
func (c *Conv1D[T]) Forward(x []T) []T {
	c.InLen = len(x)
	outPer := outputLen(c.InLen, c.KernelSize, c.Stride, c.Padding)
	if outPer == 0 {
		c.lastInput = nil
		c.lastOutput = nil
		return []T{}
	}
	// Save a defensive copy of the input — the upstream caller may reuse
	// its buffer (CONV-3 determinism). Same for the output.
	c.lastInput = append(c.lastInput[:0], x...)
	out := make([]T, outPer*c.NumFilters)

	var padL int
	if c.Padding == PadSame {
		padL, _ = padSamePadding(c.InLen, c.KernelSize, c.Stride)
	}

	for f := 0; f < c.NumFilters; f++ {
		wBase := f * c.KernelSize
		oBase := f * outPer
		for i := range outPer {
			start := i*c.Stride - padL
			var acc T
			for k := 0; k < c.KernelSize; k++ {
				pos := start + k
				if pos < 0 || pos >= c.InLen {
					continue // implicit zero-pad
				}
				acc += x[pos] * c.Weights[wBase+k]
			}
			if c.UseBias {
				acc += c.Biases[f]
			}
			out[oBase+i] = acc
		}
	}
	c.lastOutput = out
	return out
}

// Backward computes ∂L/∂X given ∂L/∂Y, and accumulates ∂L/∂W and ∂L/∂B in
// the internal grad buffers reachable via [Conv1D.GradSlots]. The grad
// buffers are zero-allocated lazily; subsequent calls accumulate (caller
// resets them after the optimizer step).
//
// AI-Meta:
//   - Purpose: Compute input-side gradient and accumulate weight/bias gradients per CONV-4.
//   - Usage: gradX := c.Backward(upstream).
//   - Concurrency: NotSafe; mutates gradW, gradB, gradX, lastInput / lastOutput must be the most recent Forward result.
//   - Related: [Conv1D.Forward], [Conv1D.GradSlots].
//   - Stability: Stable.
func (c *Conv1D[T]) Backward(upstream []T) []T {
	outPer := outputLen(c.InLen, c.KernelSize, c.Stride, c.Padding)
	if outPer == 0 || c.lastInput == nil {
		return nil
	}
	if len(upstream) != outPer*c.NumFilters {
		// Shape contract violation — return an empty gradient rather than
		// panic; callers can detect via len(gradX) == 0.
		return nil
	}
	if cap(c.gradW) < len(c.Weights) {
		c.gradW = make([]T, len(c.Weights))
	} else {
		c.gradW = c.gradW[:len(c.Weights)]
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
	if cap(c.gradX) < c.InLen {
		c.gradX = make([]T, c.InLen)
	} else {
		c.gradX = c.gradX[:c.InLen]
		for i := range c.gradX {
			c.gradX[i] = 0
		}
	}

	var padL int
	if c.Padding == PadSame {
		padL, _ = padSamePadding(c.InLen, c.KernelSize, c.Stride)
	}

	for f := 0; f < c.NumFilters; f++ {
		wBase := f * c.KernelSize
		oBase := f * outPer
		for i := range outPer {
			up := upstream[oBase+i]
			if c.UseBias {
				c.gradB[f] += up
			}
			start := i*c.Stride - padL
			for k := 0; k < c.KernelSize; k++ {
				pos := start + k
				if pos < 0 || pos >= c.InLen {
					continue
				}
				c.gradW[wBase+k] += c.lastInput[pos] * up
				c.gradX[pos] += c.Weights[wBase+k] * up
			}
		}
	}
	return c.gradX
}

// GradSlots returns the gradient accumulators for the optimizer Step()
// call. Both slices share storage with the layer; the caller MUST NOT
// retain references past the next Forward / Backward pair.
func (c *Conv1D[T]) GradSlots() (gradW, gradB []T) {
	return c.gradW, c.gradB
}

// ApplyGradSGD applies an in-place SGD step to the kernel weights (and biases
// when present) from the gradients accumulated by the most recent Backward.
// This is the hook the training loop calls to update the layer.
func (c *Conv1D[T]) ApplyGradSGD(lr T) {
	applyConvGrad(c.Weights, c.gradW, lr)
	if c.UseBias {
		applyConvGrad(c.Biases, c.gradB, lr)
	}
}

// applyConvGrad performs w[i] -= lr·g[i] over the shared prefix of w and g.
func applyConvGrad[T utils.Float](w, g []T, lr T) {
	n := min(len(w), len(g))
	for i := 0; i < n; i++ {
		w[i] -= lr * g[i]
	}
}

// Validate verifies the layer is well-formed at compile time. Returns an
// error wrapping [utils.ErrUserConfig] when shape constraints are violated
// (CONV-1, CONV-2).
//
// AI-Meta:
//   - Purpose: Validate Conv1D parameters at network compile time.
//   - Usage: if err := c.Validate(inLen); err != nil { return err }.
//   - Concurrency: Safe; read-only access to layer fields.
//   - Related: [Conv1D], [outputLen].
//   - Stability: Stable.
func (c *Conv1D[T]) Validate(inLen int) error {
	if c.NumFilters <= 0 {
		return utils.Newf(utils.ErrUserConfig, "Conv1D: numFilters must be positive, got %d", c.NumFilters)
	}
	if c.KernelSize <= 0 {
		return utils.Newf(utils.ErrUserConfig, "Conv1D: kernelSize must be positive, got %d", c.KernelSize)
	}
	if c.Stride <= 0 {
		return utils.Newf(utils.ErrUserConfig, "Conv1D: stride must be positive, got %d", c.Stride)
	}
	if c.Padding == PadValid && inLen < c.KernelSize {
		return fmt.Errorf("Conv1D: input length %d < kernel size %d under PadValid: %w",
			inLen, c.KernelSize, utils.ErrConvShapeMismatch)
	}
	if len(c.Weights) != c.NumFilters*c.KernelSize {
		return utils.NewIntegrityError("Conv1D.weights",
			fmt.Sprintf("%d", c.NumFilters*c.KernelSize),
			fmt.Sprintf("%d", len(c.Weights)))
	}
	if c.UseBias && len(c.Biases) != c.NumFilters {
		return utils.NewIntegrityError("Conv1D.biases",
			fmt.Sprintf("%d", c.NumFilters),
			fmt.Sprintf("%d", len(c.Biases)))
	}
	return nil
}

// MarshalJSON implements [json.Marshaler] for round-trip persistence
// (CONV-7). The non-exported scratch buffers are excluded.
func (c *Conv1D[T]) MarshalJSON() ([]byte, error) {
	// Encode via a struct alias so json/encoding picks up the exported
	// fields directly without recursive calls into MarshalJSON.
	type alias Conv1D[T]
	return json.Marshal((*alias)(c))
}

// UnmarshalJSON implements [json.Unmarshaler] for round-trip persistence.
func (c *Conv1D[T]) UnmarshalJSON(data []byte) error {
	type alias Conv1D[T]
	if err := json.Unmarshal(data, (*alias)(c)); err != nil {
		return utils.Wrap(utils.ErrIntegrity, err, "Conv1D.UnmarshalJSON")
	}
	if c.NumFilters <= 0 || c.KernelSize <= 0 || c.Stride <= 0 {
		return utils.Newf(utils.ErrIntegrity,
			"Conv1D.UnmarshalJSON: invalid shape numFilters=%d kernelSize=%d stride=%d",
			c.NumFilters, c.KernelSize, c.Stride)
	}
	if len(c.Weights) != c.NumFilters*c.KernelSize {
		return utils.NewIntegrityError("Conv1D.weights",
			fmt.Sprintf("%d", c.NumFilters*c.KernelSize),
			fmt.Sprintf("%d", len(c.Weights)))
	}
	return nil
}
