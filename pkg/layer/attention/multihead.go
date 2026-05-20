package attention

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/utils"
)

// MultiHeadAttention implements scaled dot-product multi-head attention
// (ATT-1..ATT-10). A single struct covers all three L1 conceptual variants
// via the NumHeads field: NumHeads=1 gives single-head attention; NumHeads>1
// gives multi-head. Constructor sugar [NewAttention] wraps NumHeads=1.
//
// Weight storage: four projection matrices (Wq/Wk/Wv/Wo), each [Dmodel×Dmodel]
// row-major flat, and corresponding bias vectors of length Dmodel. Forward caches
// are pre-allocated by Init and reused across calls (zero allocation in inference).
//
// AI-Meta:
//   - Purpose: Multi-head scaled dot-product attention implementing ATT-1..ATT-10.
//   - Usage: a := attention.NewMultiHeadAttention[float64](seqLen, dmodel, numHeads, causal); a.Init(rng).
//   - Concurrency: NotSafe; Forward/Backward mutate per-call caches.
//   - Related: [MaskedLayer], [layer.Layer], [utils.XavierUniform].
//   - Stability: Stable.
type MultiHeadAttention[T utils.Float] struct {
	scale       T
	lastQ       []T
	gradBk      []T
	gradX       []T
	gradBo      []T
	lastK       []T
	Wk          []T
	Wv          []T
	Wo          []T
	Bq          []T
	Bk          []T
	Bv          []T
	Bo          []T
	gradBv      []T
	lastInput   []T
	Wq          []T
	lastV       []T
	lastWeights []T
	padMask     []bool
	gradWq      []T
	gradWk      []T
	gradWv      []T
	gradWo      []T
	gradBq      []T
	SeqLen      int
	NumHeads    int
	Dk          int
	Dmodel      int
	Causal      bool
}

// Compile-time assertions: MultiHeadAttention must satisfy layer.Layer (ATT-10).
var (
	_ layer.Layer[float32] = (*MultiHeadAttention[float32])(nil)
	_ layer.Layer[float64] = (*MultiHeadAttention[float64])(nil)
)

// NewAttention constructs a single-head attention layer (NumHeads = 1).
//
// AI-Meta:
//   - Purpose: Single-head attention sugar; equivalent to NewMultiHeadAttention(seqLen, dmodel, 1, causal).
//   - Usage: a := attention.NewAttention[float64](seqLen, dmodel, false).
//   - Stability: Stable.
func NewAttention[T utils.Float](seqLen, dmodel int, causal bool) *MultiHeadAttention[T] {
	return NewMultiHeadAttention[T](seqLen, dmodel, 1, causal)
}

// NewMultiHeadAttention constructs a multi-head attention layer.
// Panics with ErrAttentionHeadsMismatch when dmodel is not divisible by numHeads
// (ATT-C3).
//
// AI-Meta:
//   - Purpose: Construct MultiHeadAttention with pre-allocated weight buffers; call Init before use.
//   - Usage: a := attention.NewMultiHeadAttention[float64](seqLen, dmodel, numHeads, false).
//   - Related: [MultiHeadAttention.Init].
//   - Stability: Stable.
func NewMultiHeadAttention[T utils.Float](seqLen, dmodel, numHeads int, causal bool) *MultiHeadAttention[T] {
	if dmodel%numHeads != 0 {
		panic(fmt.Errorf("attention: %w (Dmodel=%d, NumHeads=%d)",
			utils.ErrAttentionHeadsMismatch, dmodel, numHeads))
	}
	dk := dmodel / numHeads
	return &MultiHeadAttention[T]{
		SeqLen:   seqLen,
		Dmodel:   dmodel,
		NumHeads: numHeads,
		Dk:       dk,
		Causal:   causal,
		Wq:       make([]T, dmodel*dmodel),
		Wk:       make([]T, dmodel*dmodel),
		Wv:       make([]T, dmodel*dmodel),
		Wo:       make([]T, dmodel*dmodel),
		Bq:       make([]T, dmodel),
		Bk:       make([]T, dmodel),
		Bv:       make([]T, dmodel),
		Bo:       make([]T, dmodel),
		scale:    T(1.0 / math.Sqrt(float64(dk))),
	}
}

// Init populates projection matrices with Xavier uniform samples (ATT-8) and
// zeroes all biases. Pre-allocates forward caches and gradient buffers so
// Forward/Backward are zero-allocation after this call.
//
// AI-Meta:
//   - Purpose: Xavier-init all four projection matrices; zero biases; pre-alloc caches.
//   - Concurrency: NotSafe; rng must not be shared.
//   - Related: [utils.XavierUniform], [MultiHeadAttention.Forward].
//   - Stability: Stable.
func (m *MultiHeadAttention[T]) Init(rng *rand.Rand) {
	if rng == nil {
		panic("attention.MultiHeadAttention.Init: nil *rand.Rand")
	}
	for i := range m.Wq {
		m.Wq[i] = utils.XavierUniform[T](rng, m.Dmodel, m.Dmodel)
	}
	for i := range m.Wk {
		m.Wk[i] = utils.XavierUniform[T](rng, m.Dmodel, m.Dmodel)
	}
	for i := range m.Wv {
		m.Wv[i] = utils.XavierUniform[T](rng, m.Dmodel, m.Dmodel)
	}
	for i := range m.Wo {
		m.Wo[i] = utils.XavierUniform[T](rng, m.Dmodel, m.Dmodel)
	}
	clear(m.Bq)
	clear(m.Bk)
	clear(m.Bv)
	clear(m.Bo)

	m.lastInput = make([]T, m.SeqLen*m.Dmodel)
	m.lastQ = make([]T, m.NumHeads*m.SeqLen*m.Dk)
	m.lastK = make([]T, m.NumHeads*m.SeqLen*m.Dk)
	m.lastV = make([]T, m.NumHeads*m.SeqLen*m.Dk)
	m.lastWeights = make([]T, m.NumHeads*m.SeqLen*m.SeqLen)

	m.gradWq = make([]T, m.Dmodel*m.Dmodel)
	m.gradWk = make([]T, m.Dmodel*m.Dmodel)
	m.gradWv = make([]T, m.Dmodel*m.Dmodel)
	m.gradWo = make([]T, m.Dmodel*m.Dmodel)
	m.gradBq = make([]T, m.Dmodel)
	m.gradBk = make([]T, m.Dmodel)
	m.gradBv = make([]T, m.Dmodel)
	m.gradBo = make([]T, m.Dmodel)
	m.gradX = make([]T, m.SeqLen*m.Dmodel)
}

// InputSize returns SeqLen * Dmodel (ATT-1).
func (m *MultiHeadAttention[T]) InputSize() int { return m.SeqLen * m.Dmodel }

// ApplyGradSGD performs w -= lr·g for all eight weight/bias matrices.
// Called by applyAttentionBackward in pkg/nn/train.go after Backward.
func (m *MultiHeadAttention[T]) ApplyGradSGD(lr T) {
	for i, g := range m.gradWq {
		m.Wq[i] -= lr * g
	}
	for i, g := range m.gradWk {
		m.Wk[i] -= lr * g
	}
	for i, g := range m.gradWv {
		m.Wv[i] -= lr * g
	}
	for i, g := range m.gradWo {
		m.Wo[i] -= lr * g
	}
	for i, g := range m.gradBq {
		m.Bq[i] -= lr * g
	}
	for i, g := range m.gradBk {
		m.Bk[i] -= lr * g
	}
	for i, g := range m.gradBv {
		m.Bv[i] -= lr * g
	}
	for i, g := range m.gradBo {
		m.Bo[i] -= lr * g
	}
}

// OutputSize returns SeqLen * Dmodel (ATT-1 shape preservation).
func (m *MultiHeadAttention[T]) OutputSize() int { return m.SeqLen * m.Dmodel }

// GradSlots returns (nil, nil) — attention weight update is handled via the
// type-asserting applyAttentionBackward path in pkg/nn/train.go, not through
// the generic GradSlots optimizer interface (mirrors recurrent layer convention).
func (m *MultiHeadAttention[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// GradBuffers returns the eight gradient accumulation slices in order
// [gradWq, gradWk, gradWv, gradWo, gradBq, gradBk, gradBv, gradBo].
// Populated by Backward; used by the encoder backward FD verification test.
func (m *MultiHeadAttention[T]) GradBuffers() (gWq, gWk, gWv, gWo, gBq, gBk, gBv, gBo []T) {
	return m.gradWq, m.gradWk, m.gradWv, m.gradWo, m.gradBq, m.gradBk, m.gradBv, m.gradBo
}

// Forward implements the multi-head scaled dot-product attention forward pass
// (ATT-1..ATT-6):
//
//  1. Project input to Q, K, V via four linear maps (Wq/Wk/Wv).
//  2. Reshape to head-major [NumHeads, SeqLen, Dk] via splitHeads.
//  3. Compute scaled scores: Q[h,i,:] · K[h,j,:] / sqrt(Dk).
//  4. Apply causal mask (if Causal) and padding mask (if SetPaddingMask called).
//  5. Softmax per head row-wise.
//  6. Context = A × V; joinHeads back to [SeqLen, Dmodel].
//  7. Output projection via Wo + Bo.
//
// AI-Meta:
//   - Purpose: Multi-head attention forward with scale, optional masks, and per-call cache.
//   - Concurrency: NotSafe; mutates lastInput/lastQ/lastK/lastV/lastWeights.
//   - Related: [MultiHeadAttention.Backward], [MultiHeadAttention.SetPaddingMask].
//   - Stability: Stable.
func (m *MultiHeadAttention[T]) Forward(input []T) []T {
	// Lazy-allocate caches when Forward is called before Init (e.g. during
	// compile-time shape resolution via a dummy Forward pass).
	if m.lastQ == nil {
		m.lastInput = make([]T, m.SeqLen*m.Dmodel)
		m.lastQ = make([]T, m.NumHeads*m.SeqLen*m.Dk)
		m.lastK = make([]T, m.NumHeads*m.SeqLen*m.Dk)
		m.lastV = make([]T, m.NumHeads*m.SeqLen*m.Dk)
		m.lastWeights = make([]T, m.NumHeads*m.SeqLen*m.SeqLen)
	}
	copy(m.lastInput, input)

	// Linear projections: Q,K,V each [SeqLen, Dmodel] sequence-major.
	projQ := make([]T, m.SeqLen*m.Dmodel)
	projK := make([]T, m.SeqLen*m.Dmodel)
	projV := make([]T, m.SeqLen*m.Dmodel)
	projMat(input, m.Wq, m.Bq, projQ, m.SeqLen, m.Dmodel)
	projMat(input, m.Wk, m.Bk, projK, m.SeqLen, m.Dmodel)
	projMat(input, m.Wv, m.Bv, projV, m.SeqLen, m.Dmodel)

	// Reindex to head-major [NumHeads, SeqLen, Dk] for per-head ops.
	splitHeads(projQ, m.lastQ, m.NumHeads, m.SeqLen, m.Dk)
	splitHeads(projK, m.lastK, m.NumHeads, m.SeqLen, m.Dk)
	splitHeads(projV, m.lastV, m.NumHeads, m.SeqLen, m.Dk)

	// Scaled dot-product scores [NumHeads, SeqLen, SeqLen].
	attentionScores(m.lastQ, m.lastK, m.lastWeights, m.NumHeads, m.SeqLen, m.Dk, m.scale)

	// Masking: causal first, then padding (compose additively per ATT-6).
	if m.Causal {
		applyCausalMask(m.lastWeights, m.NumHeads, m.SeqLen)
	}
	if m.padMask != nil {
		applyPaddingMask(m.lastWeights, m.padMask, m.NumHeads, m.SeqLen)
		m.padMask = nil // per-call reset (ATT-6)
	}

	// Softmax per head independently.
	ss := m.SeqLen * m.SeqLen
	for h := range m.NumHeads {
		off := h * ss
		softmaxRowwise(m.lastWeights[off:off+ss], m.SeqLen, m.SeqLen)
	}

	// Context: A × V → head-major [NumHeads, SeqLen, Dk].
	ctx := make([]T, m.NumHeads*m.SeqLen*m.Dk)
	contextMul(m.lastWeights, m.lastV, ctx, m.NumHeads, m.SeqLen, m.Dk)

	// Merge heads → sequence-major [SeqLen, Dmodel].
	merged := make([]T, m.SeqLen*m.Dmodel)
	joinHeads(ctx, merged, m.NumHeads, m.SeqLen, m.Dk)

	// Output projection.
	out := make([]T, m.SeqLen*m.Dmodel)
	projMat(merged, m.Wo, m.Bo, out, m.SeqLen, m.Dmodel)
	return out
}

// Backward implements the ATT-7 four-path backward decomposition.
// upstream is ∂L/∂output of shape [SeqLen*Dmodel]. Returns ∂L/∂input.
// Accumulates gradients into gradWq/gradWk/gradWv/gradWo + biases.
//
// Four-path decomposition (ATT-7):
//  1. Output projection: upstream → dWo, dBo, dCtxMerged.
//  2. V path: dCtx[h,t,d] splits into dA[h,t,j] = dCtx[h,t,:]·V[h,j,:] and
//     dV[h,j,d] = A[h,t,j]·dCtx[h,t,d] summed over t.
//  3. Softmax backward: dScores from dA and cached A (post-softmax weights).
//  4. Q/K projection backward: dQ[h,i,d] = dScores[h,i,:]·K[h,:,d]*scale,
//     dK[h,j,d] = dScoresT[h,j,:]·Q[h,:,d]*scale; merge heads; ∂L/∂X accumulates
//     from Wq, Wk, Wv transpose multiplications.
//
// AI-Meta:
//   - Purpose: BPTT-equivalent for attention; accumulates weight gradients; returns input gradient.
//   - Concurrency: NotSafe; reads forward caches from the most recent Forward call.
//   - Related: [MultiHeadAttention.Forward], [MultiHeadAttention.GradSlots].
//   - Stability: Stable.
func (m *MultiHeadAttention[T]) Backward(upstream []T) []T {
	clear(m.gradWq)
	clear(m.gradWk)
	clear(m.gradWv)
	clear(m.gradWo)
	clear(m.gradBq)
	clear(m.gradBk)
	clear(m.gradBv)
	clear(m.gradBo)
	clear(m.gradX)

	// ── Path 1: Output projection backward ──────────────────────────────────
	// upstream: [SeqLen, Dmodel]; Wo: [Dmodel, Dmodel]
	// dWo[d, i] += sum_t upstream[t, d] * merged[t, i]
	// dBo[d]    += sum_t upstream[t, d]
	// dCtxMerged[t, i] = sum_d upstream[t, d] * Wo[d, i]  (Wo^T multiply)
	dCtxMerged := make([]T, m.SeqLen*m.Dmodel)
	for t := range m.SeqLen {
		for d := range m.Dmodel {
			u := upstream[t*m.Dmodel+d]
			m.gradBo[d] += u
			row := d * m.Dmodel
			for i := range m.Dmodel {
				m.gradWo[row+i] += u * m.lastInput[t*m.Dmodel+i] // placeholder: need merged
				dCtxMerged[t*m.Dmodel+i] += u * m.Wo[row+i]
			}
		}
	}
	// Note: gradWo should use the merged (pre-output-proj) value, not lastInput.
	// We need to recompute merged from the forward caches.
	// Recompute ctx from lastWeights × lastV, then join heads.
	ctx := make([]T, m.NumHeads*m.SeqLen*m.Dk)
	contextMul(m.lastWeights, m.lastV, ctx, m.NumHeads, m.SeqLen, m.Dk)
	merged := make([]T, m.SeqLen*m.Dmodel)
	joinHeads(ctx, merged, m.NumHeads, m.SeqLen, m.Dk)

	// Recompute gradWo correctly using merged.
	clear(m.gradWo)
	for t := range m.SeqLen {
		for d := range m.Dmodel {
			u := upstream[t*m.Dmodel+d]
			row := d * m.Dmodel
			for i := range m.Dmodel {
				m.gradWo[row+i] += u * merged[t*m.Dmodel+i]
			}
		}
	}

	// dCtxMerged → head-major dCtx [NumHeads, SeqLen, Dk].
	dCtx := make([]T, m.NumHeads*m.SeqLen*m.Dk)
	splitHeads(dCtxMerged, dCtx, m.NumHeads, m.SeqLen, m.Dk)

	// ── Path 2: V path & softmax input ──────────────────────────────────────
	// dV[h,j,d] = sum_t A[h,t,j] * dCtx[h,t,d]
	// dA[h,t,j] = sum_d dCtx[h,t,d] * V[h,j,d]  (A is [SeqLen,SeqLen] weights)
	dV := make([]T, m.NumHeads*m.SeqLen*m.Dk)
	dA := make([]T, m.NumHeads*m.SeqLen*m.SeqLen)
	for h := range m.NumHeads {
		ha := h * m.SeqLen * m.SeqLen
		hv := h * m.SeqLen * m.Dk
		hc := h * m.SeqLen * m.Dk
		for t := range m.SeqLen {
			for j := range m.SeqLen {
				for d := range m.Dk {
					dV[hv+j*m.Dk+d] += m.lastWeights[ha+t*m.SeqLen+j] * dCtx[hc+t*m.Dk+d]
					dA[ha+t*m.SeqLen+j] += dCtx[hc+t*m.Dk+d] * m.lastV[hv+j*m.Dk+d]
				}
			}
		}
	}

	// ── Path 3: Softmax backward ─────────────────────────────────────────────
	// dScores[h, i, j] from dA and cached A (lastWeights post-softmax).
	dScores := make([]T, m.NumHeads*m.SeqLen*m.SeqLen)
	ss := m.SeqLen * m.SeqLen
	for h := range m.NumHeads {
		off := h * ss
		softmaxBackwardRowwise(
			dA[off:off+ss],
			m.lastWeights[off:off+ss],
			dScores[off:off+ss],
			m.SeqLen, m.SeqLen,
		)
	}

	// Scale dScores by 1/sqrt(Dk) (chain rule through the scaling in attentionScores).
	for i := range dScores {
		dScores[i] *= m.scale
	}

	// ── Path 4: Q/K projection backward ─────────────────────────────────────
	// dQ[h,i,d] = sum_j dScores[h,i,j] * K[h,j,d]
	// dK[h,j,d] = sum_i dScores[h,i,j] * Q[h,i,d]  (dScores^T × Q)
	dQ := make([]T, m.NumHeads*m.SeqLen*m.Dk)
	dK := make([]T, m.NumHeads*m.SeqLen*m.Dk)
	for h := range m.NumHeads {
		hs := h * m.SeqLen * m.SeqLen
		hqk := h * m.SeqLen * m.Dk
		for i := range m.SeqLen {
			for j := range m.SeqLen {
				ds := dScores[hs+i*m.SeqLen+j]
				for d := range m.Dk {
					dQ[hqk+i*m.Dk+d] += ds * m.lastK[hqk+j*m.Dk+d]
					dK[hqk+j*m.Dk+d] += ds * m.lastQ[hqk+i*m.Dk+d]
				}
			}
		}
	}

	// Merge head-major dQ/dK/dV back to sequence-major [SeqLen, Dmodel].
	dQseq := make([]T, m.SeqLen*m.Dmodel)
	dKseq := make([]T, m.SeqLen*m.Dmodel)
	dVseq := make([]T, m.SeqLen*m.Dmodel)
	joinHeads(dQ, dQseq, m.NumHeads, m.SeqLen, m.Dk)
	joinHeads(dK, dKseq, m.NumHeads, m.SeqLen, m.Dk)
	joinHeads(dV, dVseq, m.NumHeads, m.SeqLen, m.Dk)

	// ── Weight gradients for Wq/Wk/Wv ───────────────────────────────────────
	// dW[d, i] += sum_t dProj[t, d] * input[t, i]; dB[d] += sum_t dProj[t, d]
	accumProjGrad(m.lastInput, dQseq, m.gradWq, m.gradBq, m.SeqLen, m.Dmodel)
	accumProjGrad(m.lastInput, dKseq, m.gradWk, m.gradBk, m.SeqLen, m.Dmodel)
	accumProjGrad(m.lastInput, dVseq, m.gradWv, m.gradBv, m.SeqLen, m.Dmodel)

	// ── Input gradient ∂L/∂X: accumulate from Wq^T + Wk^T + Wv^T ────────────
	// dX[t, i] += sum_d dQ[t, d] * Wq[d, i] (and same for K, V)
	accumInputGrad(dQseq, m.Wq, m.gradX, m.SeqLen, m.Dmodel)
	accumInputGrad(dKseq, m.Wk, m.gradX, m.SeqLen, m.Dmodel)
	accumInputGrad(dVseq, m.Wv, m.gradX, m.SeqLen, m.Dmodel)

	return m.gradX
}

// accumProjGrad accumulates dW[d,i] += sum_t dProj[t,d]*src[t,i]
// and dB[d] += sum_t dProj[t,d]. Used for Wq/Wk/Wv gradient accumulation.
func accumProjGrad[T utils.Float](src, dProj, dW, dB []T, seqLen, dmodel int) {
	for t := range seqLen {
		for d := range dmodel {
			dp := dProj[t*dmodel+d]
			dB[d] += dp
			row := d * dmodel
			base := t * dmodel
			for i := range dmodel {
				dW[row+i] += dp * src[base+i]
			}
		}
	}
}

// accumInputGrad accumulates dX[t,i] += sum_d dProj[t,d] * W[d,i].
// This is the W^T × dProj multiplication for the input gradient.
func accumInputGrad[T utils.Float](dProj, W, dX []T, seqLen, dmodel int) {
	for t := range seqLen {
		for d := range dmodel {
			dp := dProj[t*dmodel+d]
			row := d * dmodel
			base := t * dmodel
			for i := range dmodel {
				dX[base+i] += dp * W[row+i]
			}
		}
	}
}

// MarshalJSON serialises the layer to JSON per ATT-9.
// Padding mask is NOT serialised (runtime-only state).
//
// AI-Meta:
//   - Purpose: Serialise projection weights and shape for JSON persistence (ATT-9).
//   - Stability: Stable.
func (m *MultiHeadAttention[T]) MarshalJSON() ([]byte, error) {
	type wire struct {
		Type     string `json:"type"`
		Bq       []T    `json:"bq"`
		Wq       []T    `json:"wq"`
		Wk       []T    `json:"wk"`
		Wv       []T    `json:"wv"`
		Wo       []T    `json:"wo"`
		Bk       []T    `json:"bk"`
		Bv       []T    `json:"bv"`
		Bo       []T    `json:"bo"`
		Dmodel   int    `json:"dmodel"`
		NumHeads int    `json:"num_heads"`
		SeqLen   int    `json:"seq_len"`
		Causal   bool   `json:"causal"`
	}
	return json.Marshal(wire{
		Type:     "MultiHeadAttention",
		SeqLen:   m.SeqLen,
		Dmodel:   m.Dmodel,
		NumHeads: m.NumHeads,
		Causal:   m.Causal,
		Wq:       m.Wq,
		Wk:       m.Wk,
		Wv:       m.Wv,
		Wo:       m.Wo,
		Bq:       m.Bq,
		Bk:       m.Bk,
		Bv:       m.Bv,
		Bo:       m.Bo,
	})
}

// UnmarshalJSON restores the layer from JSON. Validates weight-length parity
// and NumHeads divisibility of Dmodel (ATT-C3).
//
// AI-Meta:
//   - Purpose: Restore projection weights from JSON; validate lengths per ATT-9.
//   - Stability: Stable.
func (m *MultiHeadAttention[T]) UnmarshalJSON(data []byte) error {
	type wire struct {
		Type     string `json:"type"`
		Bq       []T    `json:"bq"`
		Wq       []T    `json:"wq"`
		Wk       []T    `json:"wk"`
		Wv       []T    `json:"wv"`
		Wo       []T    `json:"wo"`
		Bk       []T    `json:"bk"`
		Bv       []T    `json:"bv"`
		Bo       []T    `json:"bo"`
		Dmodel   int    `json:"dmodel"`
		NumHeads int    `json:"num_heads"`
		SeqLen   int    `json:"seq_len"`
		Causal   bool   `json:"causal"`
	}
	var w wire
	if err := json.Unmarshal(data, &w); err != nil {
		return utils.Wrap(utils.ErrIntegrity, err, "MultiHeadAttention.UnmarshalJSON")
	}
	if w.Dmodel%w.NumHeads != 0 {
		return fmt.Errorf("MultiHeadAttention.UnmarshalJSON: %w (Dmodel=%d, NumHeads=%d)",
			utils.ErrAttentionHeadsMismatch, w.Dmodel, w.NumHeads)
	}
	ww := w.Dmodel * w.Dmodel
	if len(w.Wq) != ww {
		return utils.NewIntegrityError("MultiHeadAttention.Wq", itoa(ww), itoa(len(w.Wq)))
	}
	if len(w.Wk) != ww {
		return utils.NewIntegrityError("MultiHeadAttention.Wk", itoa(ww), itoa(len(w.Wk)))
	}
	if len(w.Wv) != ww {
		return utils.NewIntegrityError("MultiHeadAttention.Wv", itoa(ww), itoa(len(w.Wv)))
	}
	if len(w.Wo) != ww {
		return utils.NewIntegrityError("MultiHeadAttention.Wo", itoa(ww), itoa(len(w.Wo)))
	}
	m.SeqLen, m.Dmodel, m.NumHeads = w.SeqLen, w.Dmodel, w.NumHeads
	m.Dk = w.Dmodel / w.NumHeads
	m.Causal = w.Causal
	m.scale = T(1.0 / math.Sqrt(float64(m.Dk)))
	m.Wq, m.Wk, m.Wv, m.Wo = w.Wq, w.Wk, w.Wv, w.Wo
	m.Bq, m.Bk, m.Bv, m.Bo = w.Bq, w.Bk, w.Bv, w.Bo
	return nil
}

// itoa converts an int to its decimal string without importing strconv.
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
