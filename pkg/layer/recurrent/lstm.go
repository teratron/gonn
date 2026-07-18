package recurrent

import (
	"encoding/json"
	"math"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/utils"
)

// LSTM is a Long Short-Term Memory recurrent layer with fused gate matrices.
// Gate layout (leading axis): [i(input), f(forget), g(cell), o(output)].
//
// Weight storage: Wx [4*Hidden×InSize] row-major, Wh [4*Hidden×Hidden] row-major.
// The forget-gate bias B[Hidden:2*Hidden] is initialised to 1.0 (REC-3 convention).
//
// AI-Meta:
//   - Purpose: LSTM layer with 4-gate fused matmul; BPTT via cached (h,c,gate) state (REC-3, REC-5).
//   - Usage: l := recurrent.NewLSTM[float64](seqLen, inSize, hidden); l.Init(rng); y := l.Forward(x).
//   - Concurrency: NotSafe; Forward/Backward mutate per-step caches.
//   - Related: [SimpleRNN], [conv.Layer], [utils.Orthogonal].
//   - Stability: Stable.
type LSTM[T utils.Float] struct {
	gradB         []T
	gradWx        []T
	B             []T `json:"b"`
	stepC         []T
	stepH         []T
	gradX         []T
	Wh            []T `json:"wh"`
	lastGateActiv []T
	lastCell      []T
	lastHidden    []T
	lastInput     []T
	gradWh        []T
	Wx            []T `json:"wx"`
	Hidden        int `json:"hidden"`
	InSize        int `json:"in_size"`
	SeqLen        int `json:"seq_len"`
}

// Compile-time assertion: LSTM must satisfy conv.Layer (REC-9).
var (
	_ conv.Layer[float32] = (*LSTM[float32])(nil)
	_ conv.Layer[float64] = (*LSTM[float64])(nil)
)

// NewLSTM constructs an LSTM with zeroed weight buffers. Call Init to populate weights.
//
// AI-Meta:
//   - Purpose: Allocate an LSTM with zero-initialised weight buffers.
//   - Usage: l := recurrent.NewLSTM[float64](seqLen, inSize, hidden).
//   - Related: [LSTM.Init].
//   - Stability: Stable.
func NewLSTM[T utils.Float](seqLen, inSize, hidden int) *LSTM[T] {
	return &LSTM[T]{
		SeqLen: seqLen,
		InSize: inSize,
		Hidden: hidden,
		Wx:     make([]T, 4*hidden*inSize),
		Wh:     make([]T, 4*hidden*hidden),
		B:      make([]T, 4*hidden),
	}
}

// Init populates weights: Xavier uniform for Wx columns; four independent orthogonal
// blocks for Wh (one per gate, each Hidden×Hidden); zeros for B except forget-gate
// bias = 1.0 (REC-7). Pre-allocates all BPTT and gradient buffers.
//
// AI-Meta:
//   - Purpose: Sample Wx via XavierUniform, Wh via 4×Orthogonal blocks (REC-7); forget bias = 1.
//   - Concurrency: NotSafe; rng must not be shared.
//   - Related: [utils.XavierUniform], [utils.Orthogonal].
//   - Stability: Stable.
func (l *LSTM[T]) Init(rng *rand.Rand) {
	if rng == nil {
		panic("recurrent.LSTM.Init: nil *rand.Rand")
	}
	g := 4 * l.Hidden
	for i := range l.Wx {
		l.Wx[i] = utils.XavierUniform[T](rng, l.InSize, g)
	}
	// Four independent orthogonal blocks stacked along the gate axis.
	for gate := range 4 {
		q := utils.Orthogonal[T](rng, l.Hidden)
		base := gate * l.Hidden * l.Hidden
		copy(l.Wh[base:base+l.Hidden*l.Hidden], q)
	}
	clear(l.B)
	// Forget-gate bias initialised to 1.0 (encourages long-term memory at start).
	for i := l.Hidden; i < 2*l.Hidden; i++ {
		l.B[i] = 1
	}

	l.lastInput = make([]T, l.SeqLen*l.InSize)
	l.lastHidden = initHiddenCache[T](l.SeqLen, l.Hidden)
	l.lastCell = initHiddenCache[T](l.SeqLen, l.Hidden)
	l.lastGateActiv = make([]T, l.SeqLen*4*l.Hidden)
	l.gradWx = make([]T, len(l.Wx))
	l.gradWh = make([]T, len(l.Wh))
	l.gradB = make([]T, len(l.B))
	l.gradX = make([]T, l.SeqLen*l.InSize)
}

// InputSize returns the flat input length: SeqLen * InSize.
func (l *LSTM[T]) InputSize() int { return l.SeqLen * l.InSize }

// OutputSize returns the flat output length: SeqLen * Hidden.
func (l *LSTM[T]) OutputSize() int { return l.SeqLen * l.Hidden }

// GradSlots exposes the input-weight and bias gradient accumulators for the optimizer.
// Returns (gradWx, gradB); gradWh is updated alongside and accessible via GradWh.
func (l *LSTM[T]) GradSlots() (gradW, gradB []T) { return l.gradWx, l.gradB }

// GradWh returns the recurrent-weight gradient accumulator.
func (l *LSTM[T]) GradWh() []T { return l.gradWh }

// ApplyGradSGD updates all four gate weight blocks (Wx, Wh, B) in place from the
// gradients accumulated by the most recent Backward. Without this hook the LSTM
// stayed frozen during training (audit C2).
func (l *LSTM[T]) ApplyGradSGD(lr T) {
	sgdApply(l.Wx, l.gradWx, lr)
	sgdApply(l.Wh, l.gradWh, lr)
	sgdApply(l.B, l.gradB, lr)
}

// Forward computes all SeqLen (h_t, c_t) pairs and returns hidden states flat as
// [SeqLen*Hidden]. Input must be [SeqLen*InSize]. Caches activations for Backward.
//
// AI-Meta:
//   - Purpose: Run the 4-gate LSTM recurrence over SeqLen steps; cache gate/cell state for BPTT.
//   - Concurrency: NotSafe; mutates lastInput, lastHidden, lastCell, lastGateActiv.
//   - Related: [LSTM.Backward], [LSTM.Step].
//   - Stability: Stable.
func (l *LSTM[T]) Forward(input []T) []T {
	g4 := 4 * l.Hidden

	l.lastInput = resizeSlice(l.lastInput, l.SeqLen*l.InSize)
	copy(l.lastInput, input)

	l.lastHidden = resizeSlice(l.lastHidden, (l.SeqLen+1)*l.Hidden)
	l.lastCell = resizeSlice(l.lastCell, (l.SeqLen+1)*l.Hidden)
	l.lastGateActiv = resizeSlice(l.lastGateActiv, l.SeqLen*g4)

	// h_0 = c_0 = zero.
	clear(l.lastHidden[:l.Hidden])
	clear(l.lastCell[:l.Hidden])

	output := make([]T, l.SeqLen*l.Hidden)
	gates := make([]T, g4) // pre-activation buffer, reused each step

	for t := range l.SeqLen {
		xOff := t * l.InSize
		hPrevOff := t * l.Hidden
		hCurrOff := (t + 1) * l.Hidden
		gOff := t * g4

		// gates = B + Wx·x_t + Wh·h_{t-1}
		copy(gates, l.B)
		matVecAdd(gates, l.Wx, input[xOff:xOff+l.InSize], g4, l.InSize)
		matVecAdd(gates, l.Wh, l.lastHidden[hPrevOff:hPrevOff+l.Hidden], g4, l.Hidden)

		// Apply activations per gate: sigmoid(i,f,o), tanh(g).
		applySigmoidFused(gates, 0, l.Hidden)          // i gate
		applySigmoidFused(gates, l.Hidden, l.Hidden)   // f gate
		applyTanhFused(gates, 2*l.Hidden, l.Hidden)    // g gate (cell input)
		applySigmoidFused(gates, 3*l.Hidden, l.Hidden) // o gate

		// Cache post-activation gates.
		copy(l.lastGateActiv[gOff:gOff+g4], gates)

		// c_t = f ⊙ c_{t-1} + i ⊙ g
		cPrevOff := t * l.Hidden
		cCurrOff := (t + 1) * l.Hidden
		for j := range l.Hidden {
			ct := gates[l.Hidden+j]*l.lastCell[cPrevOff+j] + gates[j]*gates[2*l.Hidden+j]
			l.lastCell[cCurrOff+j] = ct
		}

		// h_t = o ⊙ tanh(c_t)
		for j := range l.Hidden {
			ht := gates[3*l.Hidden+j] * tanhT(l.lastCell[cCurrOff+j])
			l.lastHidden[hCurrOff+j] = ht
			output[t*l.Hidden+j] = ht
		}
	}
	return output
}

// Backward computes ∂L/∂X and accumulates ∂L/∂Wx, ∂L/∂Wh, ∂L/∂B.
// upstream must be [SeqLen*Hidden]. Returns ∂L/∂input of length SeqLen*InSize.
//
// AI-Meta:
//   - Purpose: BPTT walk t=SeqLen-1→0; accumulate all gate gradients; return input gradient.
//   - Concurrency: NotSafe; reads caches from most recent Forward call.
//   - Related: [LSTM.Forward], [LSTM.GradSlots].
//   - Stability: Stable.
func (l *LSTM[T]) Backward(upstream []T) []T {
	g4 := 4 * l.Hidden

	l.gradWx = zeroSlice(l.gradWx, len(l.Wx))
	l.gradWh = zeroSlice(l.gradWh, len(l.Wh))
	l.gradB = zeroSlice(l.gradB, len(l.B))
	l.gradX = zeroSlice(l.gradX, l.SeqLen*l.InSize)

	dh := make([]T, l.Hidden) // gradient flowing back through h
	dc := make([]T, l.Hidden) // gradient flowing back through c
	dgate := make([]T, g4)    // pre-activation gate gradient

	for t := l.SeqLen - 1; t >= 0; t-- {
		hPrevOff := t * l.Hidden
		hCurrOff := (t + 1) * l.Hidden
		gOff := t * g4
		xOff := t * l.InSize

		gates := l.lastGateActiv[gOff : gOff+g4]
		ct := l.lastCell[hCurrOff : hCurrOff+l.Hidden]
		cPrev := l.lastCell[hPrevOff : hPrevOff+l.Hidden]
		hPrev := l.lastHidden[hPrevOff : hPrevOff+l.Hidden]

		// Accumulate upstream into dh.
		for j := range l.Hidden {
			dh[j] += upstream[t*l.Hidden+j]
		}

		// Pointers into gates by gate index.
		iG := gates[0:l.Hidden]
		fG := gates[l.Hidden : 2*l.Hidden]
		gG := gates[2*l.Hidden : 3*l.Hidden]
		oG := gates[3*l.Hidden : 4*l.Hidden]

		// ∂L/∂o (pre-activation): dh * tanh(c_t) * o*(1-o)
		for j := range l.Hidden {
			tanhCt := tanhT(ct[j])
			dgate[3*l.Hidden+j] = dh[j] * tanhCt * oG[j] * (1 - oG[j])
			// Add to dc: dh * o * (1 - tanh(c_t)²)
			dc[j] += dh[j] * oG[j] * (1 - tanhCt*tanhCt)
		}
		// ∂L/∂i (pre-activation): dc * g * i*(1-i)
		for j := range l.Hidden {
			dgate[j] = dc[j] * gG[j] * iG[j] * (1 - iG[j])
		}
		// ∂L/∂f (pre-activation): dc * c_{t-1} * f*(1-f)
		for j := range l.Hidden {
			dgate[l.Hidden+j] = dc[j] * cPrev[j] * fG[j] * (1 - fG[j])
		}
		// ∂L/∂g (pre-activation): dc * i * (1 - g²)
		for j := range l.Hidden {
			dgate[2*l.Hidden+j] = dc[j] * iG[j] * (1 - gG[j]*gG[j])
		}

		// ∂L/∂B
		for j := range g4 {
			l.gradB[j] += dgate[j]
		}
		// ∂L/∂Wx += dgate ⊗ x_t
		for i := range g4 {
			for j := range l.InSize {
				l.gradWx[i*l.InSize+j] += dgate[i] * l.lastInput[xOff+j]
			}
		}
		// ∂L/∂Wh += dgate ⊗ h_{t-1}
		for i := range g4 {
			for j := range l.Hidden {
				l.gradWh[i*l.Hidden+j] += dgate[i] * hPrev[j]
			}
		}
		// ∂L/∂x_t = Wx^T · dgate
		for j := range l.InSize {
			for i := range g4 {
				l.gradX[xOff+j] += l.Wx[i*l.InSize+j] * dgate[i]
			}
		}
		// ∂L/∂h_{t-1} = Wh^T · dgate → becomes dh for t-1.
		for j := range l.Hidden {
			dh[j] = 0
			for i := range g4 {
				dh[j] += l.Wh[i*l.Hidden+j] * dgate[i]
			}
		}
		// ∂L/∂c_{t-1} = dc * f → becomes dc for t-1.
		for j := range l.Hidden {
			dc[j] = dc[j] * fG[j]
		}
	}
	return l.gradX
}

// Step advances the LSTM one time step using internal stepH/stepC state.
// Panics when Wx is nil (layer not initialised).
//
// AI-Meta:
//   - Purpose: Stateful single-step inference; updates stepH/stepC in-place.
//   - Usage: h := l.Step(x_t); use l.ResetState() between independent sequences.
//   - Concurrency: NotSafe; mutates stepH, stepC.
//   - Stability: Stable.
func (l *LSTM[T]) Step(x []T) []T {
	if l.Wx == nil {
		panic("recurrent.LSTM.Step: layer not initialised")
	}
	g4 := 4 * l.Hidden
	if l.stepH == nil {
		l.stepH = make([]T, l.Hidden)
		l.stepC = make([]T, l.Hidden)
	}
	gates := make([]T, g4)
	copy(gates, l.B)
	matVecAdd(gates, l.Wx, x, g4, l.InSize)
	matVecAdd(gates, l.Wh, l.stepH, g4, l.Hidden)

	applySigmoidFused(gates, 0, l.Hidden)
	applySigmoidFused(gates, l.Hidden, l.Hidden)
	applyTanhFused(gates, 2*l.Hidden, l.Hidden)
	applySigmoidFused(gates, 3*l.Hidden, l.Hidden)

	for j := range l.Hidden {
		l.stepC[j] = gates[l.Hidden+j]*l.stepC[j] + gates[j]*gates[2*l.Hidden+j]
		l.stepH[j] = gates[3*l.Hidden+j] * tanhT(l.stepC[j])
	}
	return l.stepH
}

// ResetState zeroes the internal Step() hidden and cell states for a new sequence.
//
// AI-Meta:
//   - Purpose: Zero stepH and stepC to start a new sequence during stateful inference.
//   - Stability: Stable.
func (l *LSTM[T]) ResetState() {
	if l.stepH != nil {
		clear(l.stepH)
	}
	if l.stepC != nil {
		clear(l.stepC)
	}
}

// MarshalJSON serialises the layer weights and shape for persistence (REC-8).
func (l *LSTM[T]) MarshalJSON() ([]byte, error) {
	type wire struct {
		Type   string `json:"type"`
		Wx     []T    `json:"wx"`
		Wh     []T    `json:"wh"`
		B      []T    `json:"b"`
		SeqLen int    `json:"seq_len"`
		InSize int    `json:"in_size"`
		Hidden int    `json:"hidden"`
	}
	return json.Marshal(wire{
		Type:   "LSTM",
		SeqLen: l.SeqLen,
		InSize: l.InSize,
		Hidden: l.Hidden,
		Wx:     l.Wx,
		Wh:     l.Wh,
		B:      l.B,
	})
}

// UnmarshalJSON restores the layer from JSON. Validates weight-length parity (REC-8).
func (l *LSTM[T]) UnmarshalJSON(data []byte) error {
	type wire struct {
		Type   string `json:"type"`
		Wx     []T    `json:"wx"`
		Wh     []T    `json:"wh"`
		B      []T    `json:"b"`
		SeqLen int    `json:"seq_len"`
		InSize int    `json:"in_size"`
		Hidden int    `json:"hidden"`
	}
	var w wire
	if err := json.Unmarshal(data, &w); err != nil {
		return utils.Wrap(utils.ErrIntegrity, err, "LSTM.UnmarshalJSON")
	}
	if len(w.Wx) != 4*w.Hidden*w.InSize {
		return utils.NewIntegrityError("LSTM.Wx",
			itoa(4*w.Hidden*w.InSize), itoa(len(w.Wx)))
	}
	if len(w.Wh) != 4*w.Hidden*w.Hidden {
		return utils.NewIntegrityError("LSTM.Wh",
			itoa(4*w.Hidden*w.Hidden), itoa(len(w.Wh)))
	}
	if len(w.B) != 4*w.Hidden {
		return utils.NewIntegrityError("LSTM.B",
			itoa(4*w.Hidden), itoa(len(w.B)))
	}
	l.SeqLen, l.InSize, l.Hidden = w.SeqLen, w.InSize, w.Hidden
	l.Wx, l.Wh, l.B = w.Wx, w.Wh, w.B
	return nil
}

// tanhT computes math.Tanh via float64 and converts the result back to T.
func tanhT[T utils.Float](x T) T {
	return T(math.Tanh(float64(x)))
}

// resizeSlice returns dst[:n] when cap(dst) >= n, otherwise allocates a new slice.
// Elements are NOT zeroed — callers zero before first write when required.
func resizeSlice[T utils.Float](dst []T, n int) []T {
	if cap(dst) >= n {
		return dst[:n]
	}
	return make([]T, n)
}
