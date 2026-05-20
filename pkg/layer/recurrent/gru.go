package recurrent

import (
	"encoding/json"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/utils"
)

// GRU is a Gated Recurrent Unit layer with 3 fused gate matrices.
// Gate layout (leading axis): [r(reset), z(update), n(candidate)].
//
// Weight storage: Wx [3*Hidden×InSize] row-major, Wh [3*Hidden×Hidden] row-major.
// The candidate gate's hidden weight Wh[2H:3H] multiplies r_t ⊙ h_{t-1} (REC-3).
//
// AI-Meta:
//   - Purpose: GRU layer with 3-gate fused matmul; BPTT via cached (r,z,n,rh) state (REC-3, REC-5).
//   - Usage: g := recurrent.NewGRU[float64](seqLen, inSize, hidden); g.Init(rng); y := g.Forward(x).
//   - Concurrency: NotSafe; Forward/Backward mutate per-step caches.
//   - Related: [LSTM], [SimpleRNN], [conv.Layer], [utils.Orthogonal].
//   - Stability: Stable.
type GRU[T utils.Float] struct {
	lastRH     []T
	lastHidden []T
	B          []T `json:"b"`
	stepH      []T
	gradX      []T
	gradB      []T
	Wh         []T `json:"wh"`
	gradWx     []T
	lastInput  []T
	Wx         []T `json:"wx"`
	lastGates  []T
	gradWh     []T
	Hidden     int `json:"hidden"`
	InSize     int `json:"in_size"`
	SeqLen     int `json:"seq_len"`
}

// Compile-time assertion: GRU must satisfy conv.Layer (REC-9).
var (
	_ conv.Layer[float32] = (*GRU[float32])(nil)
	_ conv.Layer[float64] = (*GRU[float64])(nil)
)

// NewGRU constructs a GRU with zeroed weight buffers. Call Init to populate weights.
//
// AI-Meta:
//   - Purpose: Allocate a GRU with zero-initialised weight buffers.
//   - Usage: g := recurrent.NewGRU[float64](seqLen, inSize, hidden).
//   - Related: [GRU.Init].
//   - Stability: Stable.
func NewGRU[T utils.Float](seqLen, inSize, hidden int) *GRU[T] {
	return &GRU[T]{
		SeqLen: seqLen,
		InSize: inSize,
		Hidden: hidden,
		Wx:     make([]T, 3*hidden*inSize),
		Wh:     make([]T, 3*hidden*hidden),
		B:      make([]T, 3*hidden),
	}
}

// Init populates weights: Xavier uniform for Wx columns; three independent orthogonal
// blocks for Wh (one per gate, each Hidden×Hidden); zeros for B (REC-7).
// Pre-allocates all BPTT and gradient buffers.
//
// AI-Meta:
//   - Purpose: Sample Wx via XavierUniform, Wh via 3×Orthogonal blocks (REC-7); zero-init B.
//   - Concurrency: NotSafe; rng must not be shared.
//   - Related: [utils.XavierUniform], [utils.Orthogonal].
//   - Stability: Stable.
func (g *GRU[T]) Init(rng *rand.Rand) {
	if rng == nil {
		panic("recurrent.GRU.Init: nil *rand.Rand")
	}
	g3 := 3 * g.Hidden
	for i := range g.Wx {
		g.Wx[i] = utils.XavierUniform[T](rng, g.InSize, g3)
	}
	// Three independent orthogonal blocks stacked along the gate axis.
	for gate := range 3 {
		q := utils.Orthogonal[T](rng, g.Hidden)
		base := gate * g.Hidden * g.Hidden
		copy(g.Wh[base:base+g.Hidden*g.Hidden], q)
	}
	clear(g.B)

	g.lastInput = make([]T, g.SeqLen*g.InSize)
	g.lastHidden = initHiddenCache[T](g.SeqLen, g.Hidden)
	g.lastGates = make([]T, g.SeqLen*g3)
	g.lastRH = make([]T, g.SeqLen*g.Hidden)
	g.gradWx = make([]T, len(g.Wx))
	g.gradWh = make([]T, len(g.Wh))
	g.gradB = make([]T, len(g.B))
	g.gradX = make([]T, g.SeqLen*g.InSize)
}

// InputSize returns the flat input length: SeqLen * InSize.
func (g *GRU[T]) InputSize() int { return g.SeqLen * g.InSize }

// OutputSize returns the flat output length: SeqLen * Hidden.
func (g *GRU[T]) OutputSize() int { return g.SeqLen * g.Hidden }

// GradSlots exposes the input-weight and bias gradient accumulators for the optimizer.
// Returns (gradWx, gradB); gradWh is accessible via GradWh.
func (g *GRU[T]) GradSlots() (gradW, gradB []T) { return g.gradWx, g.gradB }

// GradWh returns the recurrent-weight gradient accumulator.
func (g *GRU[T]) GradWh() []T { return g.gradWh }

// Forward computes all SeqLen h_t vectors and returns hidden states flat as
// [SeqLen*Hidden]. Input must be [SeqLen*InSize]. Caches activations for Backward.
//
// AI-Meta:
//   - Purpose: Run the 3-gate GRU recurrence over SeqLen steps; cache gate/rh state for BPTT.
//   - Concurrency: NotSafe; mutates lastInput, lastHidden, lastGates, lastRH.
//   - Related: [GRU.Backward], [GRU.Step].
//   - Stability: Stable.
func (g *GRU[T]) Forward(input []T) []T {
	g3 := 3 * g.Hidden

	g.lastInput = resizeSlice(g.lastInput, g.SeqLen*g.InSize)
	copy(g.lastInput, input)

	g.lastHidden = resizeSlice(g.lastHidden, (g.SeqLen+1)*g.Hidden)
	g.lastGates = resizeSlice(g.lastGates, g.SeqLen*g3)
	g.lastRH = resizeSlice(g.lastRH, g.SeqLen*g.Hidden)

	clear(g.lastHidden[:g.Hidden]) // h_0 = zero

	output := make([]T, g.SeqLen*g.Hidden)
	gatesBuf := make([]T, g3) // pre-activation scratch, reused per step

	for t := range g.SeqLen {
		xOff := t * g.InSize
		hPrevOff := t * g.Hidden
		hCurrOff := (t + 1) * g.Hidden
		gOff := t * g3
		rhOff := t * g.Hidden

		hPrev := g.lastHidden[hPrevOff : hPrevOff+g.Hidden]

		// Step 1: compute r and z gates: Wx[0:2H]·x + Wh[0:2H]·h_{t-1} + B[0:2H]
		copy(gatesBuf[:2*g.Hidden], g.B[:2*g.Hidden])
		for i := range 2 * g.Hidden {
			base := i * g.InSize
			for j := range g.InSize {
				gatesBuf[i] += g.Wx[base+j] * input[xOff+j]
			}
		}
		for i := range 2 * g.Hidden {
			base := i * g.Hidden
			for j := range g.Hidden {
				gatesBuf[i] += g.Wh[base+j] * hPrev[j]
			}
		}
		applySigmoidFused(gatesBuf, 0, g.Hidden)        // r_t
		applySigmoidFused(gatesBuf, g.Hidden, g.Hidden) // z_t

		// Step 2: compute rh = r_t ⊙ h_{t-1}
		rh := g.lastRH[rhOff : rhOff+g.Hidden]
		for j := range g.Hidden {
			rh[j] = gatesBuf[j] * hPrev[j]
		}

		// Step 3: compute n gate: Wx[2H:3H]·x + Wh[2H:3H]·rh + B[2H:3H]
		copy(gatesBuf[2*g.Hidden:], g.B[2*g.Hidden:])
		for i := range g.Hidden {
			row := (2*g.Hidden + i) * g.InSize
			for j := range g.InSize {
				gatesBuf[2*g.Hidden+i] += g.Wx[row+j] * input[xOff+j]
			}
		}
		for i := range g.Hidden {
			row := (2*g.Hidden + i) * g.Hidden
			for j := range g.Hidden {
				gatesBuf[2*g.Hidden+i] += g.Wh[row+j] * rh[j]
			}
		}
		applyTanhFused(gatesBuf, 2*g.Hidden, g.Hidden) // n_t

		// Cache post-activation gates [r, z, n].
		copy(g.lastGates[gOff:gOff+g3], gatesBuf)

		// Step 4: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ n_t
		hCurr := g.lastHidden[hCurrOff : hCurrOff+g.Hidden]
		for j := range g.Hidden {
			ht := (1-gatesBuf[g.Hidden+j])*hPrev[j] + gatesBuf[g.Hidden+j]*gatesBuf[2*g.Hidden+j]
			hCurr[j] = ht
			output[t*g.Hidden+j] = ht
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
//   - Related: [GRU.Forward], [GRU.GradSlots].
//   - Stability: Stable.
func (g *GRU[T]) Backward(upstream []T) []T {
	g3 := 3 * g.Hidden

	g.gradWx = zeroSlice(g.gradWx, len(g.Wx))
	g.gradWh = zeroSlice(g.gradWh, len(g.Wh))
	g.gradB = zeroSlice(g.gradB, len(g.B))
	g.gradX = zeroSlice(g.gradX, g.SeqLen*g.InSize)

	dh := make([]T, g.Hidden)    // gradient flowing back through h
	drRaw := make([]T, g.Hidden) // pre-sigmoid gradient for reset gate
	dzRaw := make([]T, g.Hidden) // pre-sigmoid gradient for update gate
	dnRaw := make([]T, g.Hidden) // pre-tanh gradient for candidate
	dRH := make([]T, g.Hidden)   // gradient w.r.t. r_t ⊙ h_{t-1}

	for t := g.SeqLen - 1; t >= 0; t-- {
		hPrevOff := t * g.Hidden
		gOff := t * g3
		xOff := t * g.InSize
		rhOff := t * g.Hidden

		gates := g.lastGates[gOff : gOff+g3]
		r := gates[:g.Hidden]
		z := gates[g.Hidden : 2*g.Hidden]
		n := gates[2*g.Hidden : 3*g.Hidden]
		hPrev := g.lastHidden[hPrevOff : hPrevOff+g.Hidden]
		rh := g.lastRH[rhOff : rhOff+g.Hidden]

		// Accumulate upstream into dh.
		for j := range g.Hidden {
			dh[j] += upstream[t*g.Hidden+j]
		}

		// ∂L/∂n (pre-tanh): dh * z * (1 - n²)
		for j := range g.Hidden {
			dnRaw[j] = dh[j] * z[j] * (1 - n[j]*n[j])
		}

		// ∂L/∂z (pre-sigmoid): dh * (n - hPrev) * z*(1-z)
		for j := range g.Hidden {
			dzRaw[j] = dh[j] * (n[j] - hPrev[j]) * z[j] * (1 - z[j])
		}

		// d_rh = Wh[2H:3H]^T · dnRaw (gradient w.r.t. rh = r ⊙ hPrev)
		clear(dRH)
		for j := range g.Hidden {
			for i := range g.Hidden {
				dRH[j] += g.Wh[(2*g.Hidden+i)*g.Hidden+j] * dnRaw[i]
			}
		}

		// ∂L/∂r (pre-sigmoid): d_rh * hPrev * r*(1-r)
		for j := range g.Hidden {
			drRaw[j] = dRH[j] * hPrev[j] * r[j] * (1 - r[j])
		}

		// Accumulate bias gradients.
		for j := range g.Hidden {
			g.gradB[j] += drRaw[j]
			g.gradB[g.Hidden+j] += dzRaw[j]
			g.gradB[2*g.Hidden+j] += dnRaw[j]
		}

		// Accumulate Wx gradients.
		for i := range g.Hidden {
			for j := range g.InSize {
				g.gradWx[i*g.InSize+j] += drRaw[i] * g.lastInput[xOff+j]
				g.gradWx[(g.Hidden+i)*g.InSize+j] += dzRaw[i] * g.lastInput[xOff+j]
				g.gradWx[(2*g.Hidden+i)*g.InSize+j] += dnRaw[i] * g.lastInput[xOff+j]
			}
		}

		// Accumulate Wh gradients.
		for i := range g.Hidden {
			for j := range g.Hidden {
				g.gradWh[i*g.Hidden+j] += drRaw[i] * hPrev[j]
				g.gradWh[(g.Hidden+i)*g.Hidden+j] += dzRaw[i] * hPrev[j]
				g.gradWh[(2*g.Hidden+i)*g.Hidden+j] += dnRaw[i] * rh[j]
			}
		}

		// ∂L/∂x_t = Wx[0:H]^T·drRaw + Wx[H:2H]^T·dzRaw + Wx[2H:3H]^T·dnRaw
		for j := range g.InSize {
			for i := range g.Hidden {
				g.gradX[xOff+j] += g.Wx[i*g.InSize+j]*drRaw[i] +
					g.Wx[(g.Hidden+i)*g.InSize+j]*dzRaw[i] +
					g.Wx[(2*g.Hidden+i)*g.InSize+j]*dnRaw[i]
			}
		}

		// Compute dh_prev for the next BPTT iteration (h_{t-1}).
		// dh_prev = dh*(1-z) + d_rh*r + Wh[0:H]^T·drRaw + Wh[H:2H]^T·dzRaw
		dhPrev := make([]T, g.Hidden)
		for j := range g.Hidden {
			dhPrev[j] = dh[j]*(1-z[j]) + dRH[j]*r[j]
		}
		for j := range g.Hidden {
			for i := range g.Hidden {
				dhPrev[j] += g.Wh[i*g.Hidden+j]*drRaw[i] +
					g.Wh[(g.Hidden+i)*g.Hidden+j]*dzRaw[i]
			}
		}
		copy(dh, dhPrev)
	}
	return g.gradX
}

// Step advances the GRU one time step using internal stepH state.
// Panics when Wx is nil (layer not initialised).
//
// AI-Meta:
//   - Purpose: Stateful single-step inference; updates stepH in-place.
//   - Usage: h := g.Step(x_t); use g.ResetState() between independent sequences.
//   - Concurrency: NotSafe; mutates stepH.
//   - Stability: Stable.
func (g *GRU[T]) Step(x []T) []T {
	if g.Wx == nil {
		panic("recurrent.GRU.Step: layer not initialised")
	}
	if g.stepH == nil {
		g.stepH = make([]T, g.Hidden)
	}
	g3 := 3 * g.Hidden
	gatesBuf := make([]T, g3)

	// r and z gates.
	copy(gatesBuf[:2*g.Hidden], g.B[:2*g.Hidden])
	for i := range 2 * g.Hidden {
		base := i * g.InSize
		for j := range g.InSize {
			gatesBuf[i] += g.Wx[base+j] * x[j]
		}
	}
	for i := range 2 * g.Hidden {
		base := i * g.Hidden
		for j := range g.Hidden {
			gatesBuf[i] += g.Wh[base+j] * g.stepH[j]
		}
	}
	applySigmoidFused(gatesBuf, 0, g.Hidden)
	applySigmoidFused(gatesBuf, g.Hidden, g.Hidden)

	// rh = r ⊙ h_{t-1}
	rh := make([]T, g.Hidden)
	for j := range g.Hidden {
		rh[j] = gatesBuf[j] * g.stepH[j]
	}

	// n gate.
	copy(gatesBuf[2*g.Hidden:], g.B[2*g.Hidden:])
	for i := range g.Hidden {
		row := (2*g.Hidden + i) * g.InSize
		for j := range g.InSize {
			gatesBuf[2*g.Hidden+i] += g.Wx[row+j] * x[j]
		}
	}
	for i := range g.Hidden {
		row := (2*g.Hidden + i) * g.Hidden
		for j := range g.Hidden {
			gatesBuf[2*g.Hidden+i] += g.Wh[row+j] * rh[j]
		}
	}
	applyTanhFused(gatesBuf, 2*g.Hidden, g.Hidden)

	// h_t = (1-z) ⊙ h_{t-1} + z ⊙ n_t
	for j := range g.Hidden {
		g.stepH[j] = (1-gatesBuf[g.Hidden+j])*g.stepH[j] + gatesBuf[g.Hidden+j]*gatesBuf[2*g.Hidden+j]
	}
	return g.stepH
}

// ResetState zeroes the internal Step() hidden state for a new sequence.
//
// AI-Meta:
//   - Purpose: Zero stepH to start a new sequence during stateful inference.
//   - Stability: Stable.
func (g *GRU[T]) ResetState() {
	if g.stepH != nil {
		clear(g.stepH)
	}
}

// MarshalJSON serialises the layer weights and shape for persistence (REC-8).
func (g *GRU[T]) MarshalJSON() ([]byte, error) {
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
		Type:   "GRU",
		SeqLen: g.SeqLen,
		InSize: g.InSize,
		Hidden: g.Hidden,
		Wx:     g.Wx,
		Wh:     g.Wh,
		B:      g.B,
	})
}

// UnmarshalJSON restores the layer from JSON. Validates weight-length parity (REC-8).
func (g *GRU[T]) UnmarshalJSON(data []byte) error {
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
		return utils.Wrap(utils.ErrIntegrity, err, "GRU.UnmarshalJSON")
	}
	if len(w.Wx) != 3*w.Hidden*w.InSize {
		return utils.NewIntegrityError("GRU.Wx",
			itoa(3*w.Hidden*w.InSize), itoa(len(w.Wx)))
	}
	if len(w.Wh) != 3*w.Hidden*w.Hidden {
		return utils.NewIntegrityError("GRU.Wh",
			itoa(3*w.Hidden*w.Hidden), itoa(len(w.Wh)))
	}
	if len(w.B) != 3*w.Hidden {
		return utils.NewIntegrityError("GRU.B",
			itoa(3*w.Hidden), itoa(len(w.B)))
	}
	g.SeqLen, g.InSize, g.Hidden = w.SeqLen, w.InSize, w.Hidden
	g.Wx, g.Wh, g.B = w.Wx, w.Wh, w.B
	return nil
}
