package attention

import (
	"math"
	"math/rand/v2"
	"testing"
)

// ── softmax helpers ─────────────────────────────────────────────────────────

func TestSoftmaxRowwise_RowSumsOne(t *testing.T) {
	scores := []float64{1, 2, 3, 4, 5, 6}
	softmaxRowwise(scores, 2, 3)
	for row := range 2 {
		var sum float64
		for col := range 3 {
			sum += scores[row*3+col]
		}
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("row %d: sum = %f, want 1.0", row, sum)
		}
	}
}

func TestSoftmaxRowwise_NumericalStability(t *testing.T) {
	// Large values — without max-subtract would overflow to Inf.
	scores := []float64{1000, 1001, 1002}
	softmaxRowwise(scores, 1, 3)
	var sum float64
	for _, v := range scores {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatal("softmax produced NaN or Inf on large input")
		}
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("sum = %f, want 1.0", sum)
	}
}

func TestSoftmaxRowwiseWithMask_MaskedPositionsZero(t *testing.T) {
	scores := []float64{1, 2, 3, 4}
	mask := []bool{true, false, true, false}
	softmaxRowwiseWithMask(scores, 1, 4, mask)
	// positions 1 and 3 (mask=false) must be exactly zero.
	if scores[1] != 0 {
		t.Errorf("masked position 1 = %f, want 0", scores[1])
	}
	if scores[3] != 0 {
		t.Errorf("masked position 3 = %f, want 0", scores[3])
	}
	sum := scores[0] + scores[2]
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("valid positions sum = %f, want 1.0", sum)
	}
}

func TestSoftmaxRowwiseWithMask_NilMask(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{1, 2, 3}
	softmaxRowwise(a, 1, 3)
	softmaxRowwiseWithMask(b, 1, 3, nil)
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-15 {
			t.Errorf("position %d: nil-mask=%f unmasked=%f differ", i, b[i], a[i])
		}
	}
}

// ── softmax backward ────────────────────────────────────────────────────────

func TestSoftmaxBackwardRowwise_FDCheck(t *testing.T) {
	const n, k = 2, 4
	rng := rand.New(rand.NewPCG(42, 0))
	scores := make([]float64, n*k)
	for i := range scores {
		scores[i] = rng.Float64()
	}
	// Forward.
	A := make([]float64, n*k)
	copy(A, scores)
	softmaxRowwise(A, n, k)

	// Upstream gradient (arbitrary).
	dA := make([]float64, n*k)
	for i := range dA {
		dA[i] = rng.Float64() - 0.5
	}

	// Analytic gradient.
	dScores := make([]float64, n*k)
	softmaxBackwardRowwise(dA, A, dScores, n, k)

	// FD gradient.
	const eps = 1e-5
	for idx := range scores {
		s := make([]float64, n*k)
		copy(s, scores)
		s[idx] += eps
		softmaxRowwise(s, n, k)
		var fwdPlus float64
		for i := range s {
			fwdPlus += dA[i] * s[i]
		}
		copy(s, scores)
		s[idx] -= eps
		softmaxRowwise(s, n, k)
		var fwdMinus float64
		for i := range s {
			fwdMinus += dA[i] * s[i]
		}
		fd := (fwdPlus - fwdMinus) / (2 * eps)
		if math.Abs(dScores[idx]-fd) > 1e-4 {
			t.Errorf("dScores[%d] analytic=%f FD=%f diff=%e",
				idx, dScores[idx], fd, math.Abs(dScores[idx]-fd))
		}
	}
}

// ── splitHeads / joinHeads round-trip ───────────────────────────────────────

func TestSplitJoinHeads_RoundTrip(t *testing.T) {
	const numHeads, seqLen, dk = 2, 3, 4
	dmodel := numHeads * dk
	src := make([]float64, seqLen*dmodel)
	for i := range src {
		src[i] = float64(i)
	}
	headMajor := make([]float64, numHeads*seqLen*dk)
	splitHeads(src, headMajor, numHeads, seqLen, dk)

	dst := make([]float64, seqLen*dmodel)
	joinHeads(headMajor, dst, numHeads, seqLen, dk)

	for i, v := range src {
		if v != dst[i] {
			t.Errorf("round-trip mismatch at %d: src=%f dst=%f", i, v, dst[i])
		}
	}
}

// ── projMat ─────────────────────────────────────────────────────────────────

func TestProjMat_BiasAdded(t *testing.T) {
	const seqLen, dmodel = 2, 3
	src := []float64{1, 0, 0, 0, 1, 0} // identity rows
	W := make([]float64, dmodel*dmodel)
	for d := range dmodel {
		W[d*dmodel+d] = 1.0 // identity matrix
	}
	bias := []float64{10, 20, 30}
	out := make([]float64, seqLen*dmodel)
	projMat(src, W, bias, out, seqLen, dmodel)
	// row 0: [1,0,0] @ I + [10,20,30] = [11, 20, 30]
	want := []float64{11, 20, 30, 10, 21, 30}
	for i, v := range want {
		if math.Abs(out[i]-v) > 1e-10 {
			t.Errorf("out[%d]=%f want %f", i, out[i], v)
		}
	}
}

// ── causal mask ─────────────────────────────────────────────────────────────

func TestApplyCausalMask_UpperTriangleNegInf(t *testing.T) {
	const numHeads, seqLen = 1, 3
	scores := make([]float64, numHeads*seqLen*seqLen)
	applyCausalMask(scores, numHeads, seqLen)
	negInf := math.Inf(-1)
	for i := range seqLen {
		for j := range seqLen {
			v := scores[i*seqLen+j]
			if j > i {
				if v != negInf {
					t.Errorf("scores[%d,%d]=%f want -Inf", i, j, v)
				}
			}
		}
	}
}

// ── MultiHeadAttention forward shape ────────────────────────────────────────

func newTestMHA(t *testing.T) *MultiHeadAttention[float64] {
	t.Helper()
	m := NewMultiHeadAttention[float64](4, 8, 2, false)
	rng := rand.New(rand.NewPCG(1, 2))
	m.Init(rng)
	return m
}

func TestMultiHeadAttention_ForwardShape(t *testing.T) {
	m := newTestMHA(t)
	x := make([]float64, m.InputSize())
	out := m.Forward(x)
	if len(out) != m.OutputSize() {
		t.Fatalf("Forward output len %d, want %d", len(out), m.OutputSize())
	}
}

func TestMultiHeadAttention_BackwardShape(t *testing.T) {
	m := newTestMHA(t)
	x := make([]float64, m.InputSize())
	out := m.Forward(x)
	dX := m.Backward(out)
	if len(dX) != m.InputSize() {
		t.Fatalf("Backward output len %d, want %d", len(dX), m.InputSize())
	}
}

func TestMultiHeadAttention_CausalMaskRetained(t *testing.T) {
	m := NewMultiHeadAttention[float64](4, 4, 1, true)
	rng := rand.New(rand.NewPCG(7, 8))
	m.Init(rng)
	x := make([]float64, m.InputSize())
	// Two consecutive Forward calls — causal mask must not leak between them.
	out1 := m.Forward(x)
	out2 := m.Forward(x)
	for i, v := range out1 {
		if v != out2[i] {
			// Different values mean state leaked — both inputs identical so outputs should match.
			t.Errorf("output[%d] differs between identical calls: %f vs %f", i, v, out2[i])
		}
	}
}

// ── FD gradient check on MultiHeadAttention ─────────────────────────────────

func TestMultiHeadAttention_BackwardFD(t *testing.T) {
	// Small model to keep test fast: seqLen=3, dmodel=4, numHeads=2.
	m := NewMultiHeadAttention[float64](3, 4, 2, false)
	rng := rand.New(rand.NewPCG(99, 0))
	m.Init(rng)

	x := make([]float64, m.InputSize())
	for i := range x {
		x[i] = rng.Float64() - 0.5
	}

	// Upstream gradient (random).
	upstream := make([]float64, m.OutputSize())
	for i := range upstream {
		upstream[i] = rng.Float64() - 0.5
	}

	// Analytic backward.
	m.Forward(x)
	dX := m.Backward(upstream)

	// FD check on input gradient.
	const eps = 1e-5
	maxErr := 0.0
	for idx := range x {
		xp := make([]float64, len(x))
		copy(xp, x)
		xp[idx] += eps
		outp := m.Forward(xp)
		var fp float64
		for i, u := range upstream {
			fp += u * outp[i]
		}

		xm := make([]float64, len(x))
		copy(xm, x)
		xm[idx] -= eps
		outm := m.Forward(xm)
		var fm float64
		for i, u := range upstream {
			fm += u * outm[i]
		}

		fd := (fp - fm) / (2 * eps)
		diff := math.Abs(dX[idx] - fd)
		if diff > maxErr {
			maxErr = diff
		}
	}
	if maxErr > 1e-4 {
		t.Errorf("FD gradient check failed: max_abs_err = %e (threshold 1e-4)", maxErr)
	}
}

// ── GradSlots ───────────────────────────────────────────────────────────────

func TestMultiHeadAttention_GradSlotsNil(t *testing.T) {
	m := newTestMHA(t)
	gW, gB := m.GradSlots()
	if gW != nil || gB != nil {
		t.Error("GradSlots must return (nil, nil) for MultiHeadAttention")
	}
}

// ── JSON round-trip ─────────────────────────────────────────────────────────

func TestMultiHeadAttention_JSONRoundTrip(t *testing.T) {
	m := newTestMHA(t)
	data, err := m.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	var m2 MultiHeadAttention[float64]
	if err := m2.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if m2.SeqLen != m.SeqLen || m2.Dmodel != m.Dmodel || m2.NumHeads != m.NumHeads {
		t.Errorf("shape mismatch after round-trip")
	}
	for i, w := range m.Wq {
		if w != m2.Wq[i] {
			t.Errorf("Wq[%d] mismatch after JSON round-trip", i)
			break
		}
	}
}

// ── MaskedLayer interface ────────────────────────────────────────────────────

func TestSetPaddingMask_ClearedAfterForward(t *testing.T) {
	m := newTestMHA(t)
	mask := []bool{true, true, true, false}
	m.SetPaddingMask(mask)
	if m.padMask == nil {
		t.Fatal("padMask not set after SetPaddingMask")
	}
	x := make([]float64, m.InputSize())
	m.Forward(x)
	if m.padMask != nil {
		t.Error("padMask not cleared after Forward (per-call semantics violated)")
	}
}

// ── NewAttention / NewMultiHeadAttention panics ──────────────────────────────

func TestNewMultiHeadAttention_HeadsMismatchPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for dmodel not divisible by numHeads")
		}
	}()
	_ = NewMultiHeadAttention[float64](4, 5, 2, false) // 5 % 2 != 0
}

// ── NewAttention sugar ───────────────────────────────────────────────────────

func TestNewAttention_SingleHead(t *testing.T) {
	m := NewAttention[float64](4, 8, false)
	if m.NumHeads != 1 {
		t.Errorf("NumHeads = %d, want 1", m.NumHeads)
	}
	if m.Causal {
		t.Error("Causal must be false when constructed with false")
	}
}

// ── ApplyGradSGD ─────────────────────────────────────────────────────────────

func TestMultiHeadAttention_ApplyGradSGD(t *testing.T) {
	m := newTestMHA(t)
	x := make([]float64, m.InputSize())
	rng := rand.New(rand.NewPCG(5, 6))
	for i := range x {
		x[i] = rng.Float64() - 0.5
	}
	upstream := make([]float64, m.OutputSize())
	for i := range upstream {
		upstream[i] = rng.Float64() - 0.5
	}
	m.Forward(x)
	m.Backward(upstream)

	wqBefore := make([]float64, len(m.Wq))
	copy(wqBefore, m.Wq)

	m.ApplyGradSGD(0.1)

	// At least some weights must have changed.
	changed := false
	for i := range m.Wq {
		if m.Wq[i] != wqBefore[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("ApplyGradSGD did not update Wq")
	}
}

// ── softmaxBackwardRowwiseWithMask ───────────────────────────────────────────

func TestSoftmaxBackwardRowwiseWithMask_MatchesUnmasked(t *testing.T) {
	const n, k = 1, 4
	rng := rand.New(rand.NewPCG(11, 22))
	scores := make([]float64, k)
	for i := range scores {
		scores[i] = rng.Float64()
	}
	A := make([]float64, k)
	copy(A, scores)
	softmaxRowwise(A, n, k)

	dA := make([]float64, k)
	for i := range dA {
		dA[i] = rng.Float64()
	}

	dS1 := make([]float64, k)
	dS2 := make([]float64, k)
	softmaxBackwardRowwise(dA, A, dS1, n, k)
	softmaxBackwardRowwiseWithMask(dA, A, dS2, n, k, []bool{true, true, true, true})

	for i := range dS1 {
		if math.Abs(dS1[i]-dS2[i]) > 1e-15 {
			t.Errorf("position %d: unmasked=%f masked=%f differ", i, dS1[i], dS2[i])
		}
	}
}

// ── UnmarshalJSON error paths ─────────────────────────────────────────────────

func TestMultiHeadAttention_UnmarshalJSON_BadJSON(t *testing.T) {
	var m MultiHeadAttention[float64]
	if err := m.UnmarshalJSON([]byte("not json")); err == nil {
		t.Error("expected error for bad JSON")
	}
}

func TestMultiHeadAttention_UnmarshalJSON_HeadsMismatch(t *testing.T) {
	// Valid JSON but dmodel=5, numHeads=2 — not divisible.
	data := []byte(`{"type":"MultiHeadAttention","seq_len":2,"dmodel":5,"num_heads":2,"causal":false,"wq":[],"wk":[],"wv":[],"wo":[],"bq":[],"bk":[],"bv":[],"bo":[]}`)
	var m MultiHeadAttention[float64]
	if err := m.UnmarshalJSON(data); err == nil {
		t.Error("expected error for heads mismatch")
	}
}

// ── ATT-3 entropy bound ──────────────────────────────────────────────────────

func TestSoftmaxRowwise_EntropyBound(t *testing.T) {
	// Uniform input → maximum entropy; peaked input → lower entropy.
	uniform := []float64{1, 1, 1, 1}
	softmaxRowwise(uniform, 1, 4)
	var hUniform float64
	for _, p := range uniform {
		if p > 0 {
			hUniform -= p * math.Log(p)
		}
	}

	peaked := []float64{10, 0, 0, 0}
	softmaxRowwise(peaked, 1, 4)
	var hPeaked float64
	for _, p := range peaked {
		if p > 0 {
			hPeaked -= p * math.Log(p)
		}
	}

	if hPeaked >= hUniform {
		t.Errorf("peaked entropy %f should be < uniform entropy %f", hPeaked, hUniform)
	}
}
