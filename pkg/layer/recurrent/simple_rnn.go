package recurrent

import (
	"encoding/json"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/utils"
)

// SimpleRNN is an Elman recurrent layer: h_t = tanh(Wxh·x_t + Whh·h_{t-1} + b_h).
// It satisfies [conv.Layer] so it composes uniformly with conv and dense layers.
//
// Weight storage: Wxh [Hidden×InSize] row-major, Whh [Hidden×Hidden] row-major.
// BPTT caches are allocated once by Init and reused across Forward/Backward calls.
//
// AI-Meta:
//   - Purpose: Elman (tanh) recurrent layer implementing BPTT via stored h_t cache (REC-2, REC-5).
//   - Usage: r := recurrent.NewSimpleRNN[float64](seqLen, inSize, hidden); r.Init(rng); y := r.Forward(x).
//   - Concurrency: NotSafe; Forward/Backward mutate per-step caches.
//   - Related: [LSTM], [conv.Layer], [utils.Orthogonal].
//   - Stability: Stable.
type SimpleRNN[T utils.Float] struct {
	gradWhh    []T
	Whh        []T `json:"whh"`
	Bh         []T `json:"bh"`
	lastInput  []T
	lastHidden []T
	gradWxh    []T
	Wxh        []T `json:"wxh"`
	gradBh     []T
	gradX      []T
	stepH      []T
	SeqLen     int `json:"seq_len"`
	InSize     int `json:"in_size"`
	Hidden     int `json:"hidden"`
}

// Compile-time assertion: SimpleRNN must satisfy conv.Layer (REC-9).
var (
	_ conv.Layer[float32] = (*SimpleRNN[float32])(nil)
	_ conv.Layer[float64] = (*SimpleRNN[float64])(nil)
)

// NewSimpleRNN constructs a SimpleRNN with zeroed weight buffers. Call Init to
// populate weights before use.
//
// AI-Meta:
//   - Purpose: Allocate a SimpleRNN with zero-initialised weight and bias buffers.
//   - Usage: r := recurrent.NewSimpleRNN[float64](seqLen, inSize, hidden).
//   - Related: [SimpleRNN.Init].
//   - Stability: Stable.
func NewSimpleRNN[T utils.Float](seqLen, inSize, hidden int) *SimpleRNN[T] {
	return &SimpleRNN[T]{
		SeqLen: seqLen,
		InSize: inSize,
		Hidden: hidden,
		Wxh:    make([]T, hidden*inSize),
		Whh:    make([]T, hidden*hidden),
		Bh:     make([]T, hidden),
	}
}

// Init populates weights: Xavier uniform for Wxh, orthogonal for Whh (REC-7),
// zero for Bh. Pre-allocates all BPTT and gradient buffers.
//
// AI-Meta:
//   - Purpose: Sample Wxh via XavierUniform, Whh via Orthogonal (REC-7); zero-init Bh; pre-alloc caches.
//   - Concurrency: NotSafe; rng must not be shared.
//   - Related: [utils.XavierUniform], [utils.Orthogonal].
//   - Stability: Stable.
func (r *SimpleRNN[T]) Init(rng *rand.Rand) {
	if rng == nil {
		panic("recurrent.SimpleRNN.Init: nil *rand.Rand")
	}
	for i := range r.Wxh {
		r.Wxh[i] = utils.XavierUniform[T](rng, r.InSize, r.Hidden)
	}
	copy(r.Whh, utils.Orthogonal[T](rng, r.Hidden))
	for i := range r.Bh {
		r.Bh[i] = 0
	}
	r.lastInput = make([]T, r.SeqLen*r.InSize)
	r.lastHidden = initHiddenCache[T](r.SeqLen, r.Hidden)
	r.gradWxh = make([]T, len(r.Wxh))
	r.gradWhh = make([]T, len(r.Whh))
	r.gradBh = make([]T, len(r.Bh))
	r.gradX = make([]T, r.SeqLen*r.InSize)
}

// InputSize returns the flat input length: SeqLen * InSize.
func (r *SimpleRNN[T]) InputSize() int { return r.SeqLen * r.InSize }

// OutputSize returns the flat output length: SeqLen * Hidden.
func (r *SimpleRNN[T]) OutputSize() int { return r.SeqLen * r.Hidden }

// GradSlots exposes the weight and bias gradient accumulators for the optimizer.
// Returns (gradWxh, gradBh); gradWhh is updated alongside but returned separately
// when the recurrent training path reads it directly.
func (r *SimpleRNN[T]) GradSlots() (gradW, gradB []T) { return r.gradWxh, r.gradBh }

// GradWhh returns the recurrent-weight gradient accumulator. The optimizer step
// for recurrent layers reads this in addition to GradSlots.
func (r *SimpleRNN[T]) GradWhh() []T { return r.gradWhh }

// Forward computes all SeqLen hidden states and returns them flat as [SeqLen*Hidden].
// Input must be [SeqLen*InSize]. Caches input and hidden states for Backward.
//
// AI-Meta:
//   - Purpose: Run the Elman recurrence over SeqLen steps; cache h_t for BPTT.
//   - Concurrency: NotSafe; mutates lastInput, lastHidden.
//   - Related: [SimpleRNN.Backward], [SimpleRNN.Step].
//   - Stability: Stable.
func (r *SimpleRNN[T]) Forward(input []T) []T {
	if cap(r.lastInput) < r.SeqLen*r.InSize {
		r.lastInput = make([]T, r.SeqLen*r.InSize)
	} else {
		r.lastInput = r.lastInput[:r.SeqLen*r.InSize]
	}
	copy(r.lastInput, input)

	if cap(r.lastHidden) < (r.SeqLen+1)*r.Hidden {
		r.lastHidden = make([]T, (r.SeqLen+1)*r.Hidden)
	} else {
		r.lastHidden = r.lastHidden[:(r.SeqLen+1)*r.Hidden]
	}
	// h_0 = zero vector at index 0.
	for i := range r.Hidden {
		r.lastHidden[i] = 0
	}

	output := make([]T, r.SeqLen*r.Hidden)
	preact := make([]T, r.Hidden)

	for t := range r.SeqLen {
		xOff := t * r.InSize
		hPrevOff := t * r.Hidden
		hCurrOff := (t + 1) * r.Hidden

		// pre-activation: Bh + Wxh·x_t + Whh·h_{t-1}
		copy(preact, r.Bh)
		matVecAdd(preact, r.Wxh, input[xOff:xOff+r.InSize], r.Hidden, r.InSize)
		matVecAdd(preact, r.Whh, r.lastHidden[hPrevOff:hPrevOff+r.Hidden], r.Hidden, r.Hidden)

		// h_t = tanh(pre-activation)
		applyTanhFused(preact, 0, r.Hidden)

		copy(r.lastHidden[hCurrOff:hCurrOff+r.Hidden], preact)
		copy(output[t*r.Hidden:(t+1)*r.Hidden], preact)
	}
	return output
}

// Backward computes ∂L/∂X and accumulates ∂L/∂Wxh, ∂L/∂Whh, ∂L/∂Bh.
// upstream must be [SeqLen*Hidden] (∂L/∂output from each time step).
// Returns ∂L/∂input of length SeqLen*InSize.
//
// AI-Meta:
//   - Purpose: BPTT walk t=SeqLen-1→0; accumulate weight gradients; return input gradient.
//   - Concurrency: NotSafe; reads lastInput/lastHidden from most recent Forward call.
//   - Related: [SimpleRNN.Forward], [SimpleRNN.GradSlots].
//   - Stability: Stable.
func (r *SimpleRNN[T]) Backward(upstream []T) []T {
	// Zero gradient accumulators.
	r.gradWxh = zeroSlice(r.gradWxh, len(r.Wxh))
	r.gradWhh = zeroSlice(r.gradWhh, len(r.Whh))
	r.gradBh = zeroSlice(r.gradBh, len(r.Bh))
	r.gradX = zeroSlice(r.gradX, r.SeqLen*r.InSize)

	// dh carries gradient from h_{t+1} backwards through Whh.
	dh := make([]T, r.Hidden)
	dtanh := make([]T, r.Hidden)

	for t := r.SeqLen - 1; t >= 0; t-- {
		hOff := (t + 1) * r.Hidden
		hPrevOff := t * r.Hidden
		xOff := t * r.InSize
		ht := r.lastHidden[hOff : hOff+r.Hidden]
		hPrev := r.lastHidden[hPrevOff : hPrevOff+r.Hidden]

		// Accumulate upstream gradient into dh.
		for i := range r.Hidden {
			dh[i] += upstream[t*r.Hidden+i]
		}

		// Backprop through tanh: dtanh = dh * (1 - h_t²).
		for i := range r.Hidden {
			dtanh[i] = dh[i] * (1 - ht[i]*ht[i])
		}

		// ∂L/∂Bh
		for i := range r.Hidden {
			r.gradBh[i] += dtanh[i]
		}
		// ∂L/∂Wxh += dtanh ⊗ x_t
		for i := range r.Hidden {
			for j := range r.InSize {
				r.gradWxh[i*r.InSize+j] += dtanh[i] * r.lastInput[xOff+j]
			}
		}
		// ∂L/∂Whh += dtanh ⊗ h_{t-1}
		for i := range r.Hidden {
			for j := range r.Hidden {
				r.gradWhh[i*r.Hidden+j] += dtanh[i] * hPrev[j]
			}
		}
		// ∂L/∂x_t = Wxh^T · dtanh
		for j := range r.InSize {
			for i := range r.Hidden {
				r.gradX[xOff+j] += r.Wxh[i*r.InSize+j] * dtanh[i]
			}
		}
		// ∂L/∂h_{t-1} = Whh^T · dtanh (becomes dh for next iteration).
		for j := range r.Hidden {
			dh[j] = 0
			for i := range r.Hidden {
				dh[j] += r.Whh[i*r.Hidden+j] * dtanh[i]
			}
		}
	}
	return r.gradX
}

// Step advances the SimpleRNN one time step using internal stepH state.
// Panics when Wxh is nil (layer not initialised).
//
// AI-Meta:
//   - Purpose: Stateful single-step inference; updates stepH in-place.
//   - Usage: h := r.Step(x_t); use r.ResetState() between independent sequences.
//   - Concurrency: NotSafe; mutates stepH.
//   - Stability: Stable.
func (r *SimpleRNN[T]) Step(x []T) []T {
	if r.Wxh == nil {
		panic("recurrent.SimpleRNN.Step: layer not initialised")
	}
	if r.stepH == nil {
		r.stepH = make([]T, r.Hidden)
	}
	preact := make([]T, r.Hidden)
	copy(preact, r.Bh)
	matVecAdd(preact, r.Wxh, x, r.Hidden, r.InSize)
	matVecAdd(preact, r.Whh, r.stepH, r.Hidden, r.Hidden)
	applyTanhFused(preact, 0, r.Hidden)
	copy(r.stepH, preact)
	return r.stepH
}

// ResetState zeroes the internal Step() hidden state for a new sequence.
//
// AI-Meta:
//   - Purpose: Zero stepH to start a new sequence during stateful inference.
//   - Stability: Stable.
func (r *SimpleRNN[T]) ResetState() {
	if r.stepH != nil {
		clear(r.stepH)
	}
}

// MarshalJSON serialises the layer weights and shape for persistence (REC-8).
func (r *SimpleRNN[T]) MarshalJSON() ([]byte, error) {
	type wire struct {
		Type   string `json:"type"`
		Wxh    []T    `json:"wxh"`
		Whh    []T    `json:"whh"`
		Bh     []T    `json:"bh"`
		SeqLen int    `json:"seq_len"`
		InSize int    `json:"in_size"`
		Hidden int    `json:"hidden"`
	}
	return json.Marshal(wire{
		Type:   "SimpleRNN",
		SeqLen: r.SeqLen,
		InSize: r.InSize,
		Hidden: r.Hidden,
		Wxh:    r.Wxh,
		Whh:    r.Whh,
		Bh:     r.Bh,
	})
}

// UnmarshalJSON restores the layer from JSON. Validates weight-length parity (REC-8).
func (r *SimpleRNN[T]) UnmarshalJSON(data []byte) error {
	type wire struct {
		Type   string `json:"type"`
		Wxh    []T    `json:"wxh"`
		Whh    []T    `json:"whh"`
		Bh     []T    `json:"bh"`
		SeqLen int    `json:"seq_len"`
		InSize int    `json:"in_size"`
		Hidden int    `json:"hidden"`
	}
	var w wire
	if err := json.Unmarshal(data, &w); err != nil {
		return utils.Wrap(utils.ErrIntegrity, err, "SimpleRNN.UnmarshalJSON")
	}
	if len(w.Wxh) != w.Hidden*w.InSize {
		return utils.NewIntegrityError("SimpleRNN.Wxh",
			itoa(w.Hidden*w.InSize), itoa(len(w.Wxh)))
	}
	if len(w.Whh) != w.Hidden*w.Hidden {
		return utils.NewIntegrityError("SimpleRNN.Whh",
			itoa(w.Hidden*w.Hidden), itoa(len(w.Whh)))
	}
	if len(w.Bh) != w.Hidden {
		return utils.NewIntegrityError("SimpleRNN.Bh",
			itoa(w.Hidden), itoa(len(w.Bh)))
	}
	r.SeqLen, r.InSize, r.Hidden = w.SeqLen, w.InSize, w.Hidden
	r.Wxh, r.Whh, r.Bh = w.Wxh, w.Whh, w.Bh
	return nil
}

// ApplyGradSGD applies an in-place SGD step to every trainable buffer
// (Wxh, Whh, Bh) from the gradients accumulated by the most recent Backward,
// then leaves the accumulators as-is (Backward zeroes them at entry). This is
// the hook the training loop calls to actually update the layer — without it
// the recurrent weights stayed frozen (audit C2).
func (r *SimpleRNN[T]) ApplyGradSGD(lr T) {
	sgdApply(r.Wxh, r.gradWxh, lr)
	sgdApply(r.Whh, r.gradWhh, lr)
	sgdApply(r.Bh, r.gradBh, lr)
}

// sgdApply performs w[i] -= lr·g[i] over the shared prefix of w and g.
func sgdApply[T utils.Float](w, g []T, lr T) {
	n := min(len(w), len(g))
	for i := 0; i < n; i++ {
		w[i] -= lr * g[i]
	}
}

// zeroSlice returns a slice of length n backed by cap(dst) when possible;
// otherwise allocates. All elements are zeroed.
func zeroSlice[T utils.Float](dst []T, n int) []T {
	if cap(dst) >= n {
		dst = dst[:n]
		clear(dst)
		return dst
	}
	return make([]T, n)
}

// itoa converts an int to its decimal string representation without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	buf := [20]byte{}
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
