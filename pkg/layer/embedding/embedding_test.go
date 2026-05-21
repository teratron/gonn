package embedding

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"testing"
)

// ── sparseGrad ──────────────────────────────────────────────────────────────

func TestSparseGrad_AddAndIter(t *testing.T) {
	sg := newSparseGrad[float64](10, 3)
	sg.add(0, []float64{1, 2, 3})
	sg.add(5, []float64{4, 5, 6})
	sg.add(0, []float64{10, 20, 30}) // accumulate same row

	visited := map[int][]float64{}
	sg.iter(func(row int, grad []float64) {
		dst := make([]float64, len(grad))
		copy(dst, grad)
		visited[row] = dst
	})

	if len(visited) != 2 {
		t.Fatalf("iter visited %d rows, want 2", len(visited))
	}
	want0 := []float64{11, 22, 33}
	for i, v := range want0 {
		if visited[0][i] != v {
			t.Errorf("row 0 grad[%d]=%f want %f", i, visited[0][i], v)
		}
	}
	want5 := []float64{4, 5, 6}
	for i, v := range want5 {
		if visited[5][i] != v {
			t.Errorf("row 5 grad[%d]=%f want %f", i, visited[5][i], v)
		}
	}
}

func TestSparseGrad_Reset(t *testing.T) {
	sg := newSparseGrad[float64](64, 4)
	sg.add(0, []float64{1, 2, 3, 4})
	sg.add(63, []float64{5, 6, 7, 8})
	sg.reset()
	if sg.NumSet != 0 {
		t.Errorf("NumSet = %d after reset, want 0", sg.NumSet)
	}
	count := 0
	sg.iter(func(_ int, _ []float64) { count++ })
	if count != 0 {
		t.Errorf("iter visited %d rows after reset, want 0", count)
	}
	// Buf must be zeroed for touched rows.
	if sg.Buf[0] != 0 || sg.Buf[63*4] != 0 {
		t.Error("Buf not zeroed after reset")
	}
}

func TestSparseGrad_NumSet(t *testing.T) {
	sg := newSparseGrad[float64](100, 2)
	sg.add(10, []float64{1, 2})
	sg.add(10, []float64{3, 4}) // same row, should not increment NumSet
	sg.add(20, []float64{5, 6})
	if sg.NumSet != 2 {
		t.Errorf("NumSet = %d, want 2", sg.NumSet)
	}
}

// ── buildSinusoidalTable ────────────────────────────────────────────────────

func TestBuildSinusoidalTable_FormulaCheck(t *testing.T) {
	const seqLen, dmodel = 4, 8
	table := buildSinusoidalTable[float64](seqLen, dmodel)
	for pos := range seqLen {
		for i := 0; i < dmodel; i += 2 {
			angle := float64(pos) / math.Pow(10000.0, float64(i)/float64(dmodel))
			wantSin := math.Sin(angle)
			got := table[pos*dmodel+i]
			if math.Abs(float64(got)-wantSin) > 1e-12 {
				t.Errorf("PE(%d,%d) sin=%f want %f", pos, i, got, wantSin)
			}
			if i+1 < dmodel {
				wantCos := math.Cos(angle)
				got = table[pos*dmodel+i+1]
				if math.Abs(float64(got)-wantCos) > 1e-12 {
					t.Errorf("PE(%d,%d) cos=%f want %f", pos, i+1, got, wantCos)
				}
			}
		}
	}
}

func TestBuildSinusoidalTable_Bounds(t *testing.T) {
	table := buildSinusoidalTable[float64](8, 16)
	for i, v := range table {
		if math.Abs(float64(v)) > 1.0+1e-10 {
			t.Errorf("table[%d]=%f out of [-1,1]", i, v)
		}
	}
}

// ── TokenEmbedding ──────────────────────────────────────────────────────────

func newTestTokenEmbedding(t *testing.T) *TokenEmbedding[float64] {
	t.Helper()
	te := NewTokenEmbedding[float64](10, 4, 8)
	rng := rand.New(rand.NewPCG(1, 0))
	te.Init(rng)
	return te
}

func TestTokenEmbedding_ForwardIDsShape(t *testing.T) {
	te := newTestTokenEmbedding(t)
	ids := []int{0, 1, 2, 3}
	out, err := te.ForwardIDs(ids)
	if err != nil {
		t.Fatalf("ForwardIDs error: %v", err)
	}
	if len(out) != te.OutputSize() {
		t.Errorf("output len %d, want %d", len(out), te.OutputSize())
	}
}

func TestTokenEmbedding_ForwardIDsOOB(t *testing.T) {
	te := newTestTokenEmbedding(t)
	_, err := te.ForwardIDs([]int{0, 10}) // id=10 >= VocabSize=10
	if err == nil {
		t.Error("expected ErrVocabOutOfRange for OOB id")
	}
}

func TestTokenEmbedding_ForwardIDsLookup(t *testing.T) {
	te := newTestTokenEmbedding(t)
	out, _ := te.ForwardIDs([]int{3, 3}) // same id twice — rows must match
	row0 := out[:te.Dmodel]
	row1 := out[te.Dmodel : 2*te.Dmodel]
	for i := range row0 {
		if row0[i] != row1[i] {
			t.Errorf("same ID produced different rows: row0[%d]=%f row1=%f", i, row0[i], row1[i])
		}
	}
}

func TestTokenEmbedding_BackwardSparseGrad(t *testing.T) {
	// SeqLen=2 so all lastIDs slots are filled by the ForwardIDs call.
	te := NewTokenEmbedding[float64](10, 2, 8)
	rng := rand.New(rand.NewPCG(1, 0))
	te.Init(rng)

	ids := []int{2, 5}
	if _, err := te.ForwardIDs(ids); err != nil {
		t.Fatal(err)
	}
	upstream := make([]float64, te.OutputSize())
	for i := range upstream {
		upstream[i] = 1.0
	}
	result := te.Backward(upstream)
	if result != nil {
		t.Error("Backward must return nil for embedding (discrete input)")
	}
	// Exactly rows 2 and 5 must be touched.
	if te.gradTable.NumSet != 2 {
		t.Errorf("NumSet = %d after backward on 2 unique IDs, want 2", te.gradTable.NumSet)
	}
}

func TestTokenEmbedding_ApplyGradSGD(t *testing.T) {
	te := newTestTokenEmbedding(t)
	if _, err := te.ForwardIDs([]int{0, 1}); err != nil {
		t.Fatal(err)
	}
	upstream := make([]float64, te.OutputSize())
	for i := range upstream {
		upstream[i] = 0.5
	}
	te.Backward(upstream)

	tableBefore := make([]float64, len(te.Table))
	copy(tableBefore, te.Table)
	te.ApplyGradSGD(0.1)

	// Rows 0 and 1 must have changed; others must not.
	changed0 := false
	for d := range te.Dmodel {
		if te.Table[d] != tableBefore[d] {
			changed0 = true
		}
	}
	unchanged2 := true
	for d := range te.Dmodel {
		if te.Table[2*te.Dmodel+d] != tableBefore[2*te.Dmodel+d] {
			unchanged2 = false
		}
	}
	if !changed0 {
		t.Error("row 0 not updated after ApplyGradSGD")
	}
	if !unchanged2 {
		t.Error("row 2 was updated but should not have been touched")
	}
	// After ApplyGradSGD gradient must be reset.
	if te.gradTable.NumSet != 0 {
		t.Error("gradient not reset after ApplyGradSGD")
	}
}

func TestTokenEmbedding_Forward_FloatIDs(t *testing.T) {
	te := newTestTokenEmbedding(t)
	x := []float64{0, 1, 2, 3}
	out := te.Forward(x)
	if len(out) != te.OutputSize() {
		t.Errorf("Forward output len %d, want %d", len(out), te.OutputSize())
	}
}

func TestTokenEmbedding_JSONRoundTrip(t *testing.T) {
	te := newTestTokenEmbedding(t)
	data, err := json.Marshal(te)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	var te2 TokenEmbedding[float64]
	if err := json.Unmarshal(data, &te2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if te2.VocabSize != te.VocabSize || te2.Dmodel != te.Dmodel {
		t.Error("shape mismatch after JSON round-trip")
	}
}

// ── PositionalEncoding ──────────────────────────────────────────────────────

func TestPositionalEncoding_Sinusoidal_ForwardAddsTable(t *testing.T) {
	pe := NewPositionalEncoding[float64](4, 8, Sinusoidal)
	rng := rand.New(rand.NewPCG(2, 0))
	pe.Init(rng)

	x := make([]float64, pe.InputSize())
	out := pe.Forward(x)
	// Zero input → output equals the sinusoidal table.
	for i, v := range out {
		if math.Abs(v-float64(pe.Table[i])) > 1e-12 {
			t.Errorf("out[%d]=%f want table[%d]=%f", i, v, i, pe.Table[i])
		}
	}
}

func TestPositionalEncoding_Sinusoidal_BackwardPassThrough(t *testing.T) {
	pe := NewPositionalEncoding[float64](4, 8, Sinusoidal)
	rng := rand.New(rand.NewPCG(3, 0))
	pe.Init(rng)
	upstream := make([]float64, pe.OutputSize())
	for i := range upstream {
		upstream[i] = float64(i + 1)
	}
	pe.Forward(make([]float64, pe.InputSize()))
	dIn := pe.Backward(upstream)
	for i, v := range upstream {
		if dIn[i] != v {
			t.Errorf("sinusoidal backward dIn[%d]=%f want %f", i, dIn[i], v)
		}
	}
}

func TestPositionalEncoding_Learnable_GradAccumulated(t *testing.T) {
	pe := NewPositionalEncoding[float64](4, 8, Learnable)
	rng := rand.New(rand.NewPCG(4, 0))
	pe.Init(rng)
	upstream := make([]float64, pe.OutputSize())
	for i := range upstream {
		upstream[i] = 1.0
	}
	pe.Forward(make([]float64, pe.InputSize()))
	pe.Backward(upstream)
	gW, _ := pe.GradSlots()
	allZero := true
	for _, v := range gW {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("learnable gradTable must be non-zero after backward")
	}
}

func TestPositionalEncoding_JSONRoundTrip(t *testing.T) {
	pe := NewPositionalEncoding[float64](4, 8, Sinusoidal)
	rng := rand.New(rand.NewPCG(5, 0))
	pe.Init(rng)
	data, err := json.Marshal(pe)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	var pe2 PositionalEncoding[float64]
	if err := json.Unmarshal(data, &pe2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if pe2.SeqLen != pe.SeqLen || pe2.Mode != pe.Mode {
		t.Error("shape mismatch after JSON round-trip")
	}
}

// ── EmbeddingStack ──────────────────────────────────────────────────────────

func newTestStack(t *testing.T) *EmbeddingStack[float64] {
	t.Helper()
	es := NewEmbeddingStack[float64](20, 5, 8, Sinusoidal)
	rng := rand.New(rand.NewPCG(7, 0))
	es.Init(rng)
	return es
}

func TestEmbeddingStack_ForwardIDsShape(t *testing.T) {
	es := newTestStack(t)
	ids := []int{0, 1, 2, 3, 4}
	out, err := es.ForwardIDs(ids)
	if err != nil {
		t.Fatalf("ForwardIDs error: %v", err)
	}
	if len(out) != es.OutputSize() {
		t.Errorf("output len %d, want %d", len(out), es.OutputSize())
	}
}

func TestEmbeddingStack_BackwardShape(t *testing.T) {
	es := newTestStack(t)
	if _, err := es.ForwardIDs([]int{0, 1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	upstream := make([]float64, es.OutputSize())
	result := es.Backward(upstream)
	// Backward of embedding returns nil (no gradient for discrete input).
	if result != nil {
		t.Error("EmbeddingStack.Backward must return nil")
	}
}

func TestEmbeddingStack_Forward_FloatIDs(t *testing.T) {
	es := newTestStack(t)
	x := []float64{0, 1, 2, 3, 4}
	out := es.Forward(x)
	if len(out) != es.OutputSize() {
		t.Errorf("Forward output len %d, want %d", len(out), es.OutputSize())
	}
}

func TestEmbeddingStack_JSONRoundTrip(t *testing.T) {
	es := newTestStack(t)
	data, err := json.Marshal(es)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	var es2 EmbeddingStack[float64]
	if err := json.Unmarshal(data, &es2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if es2.Token.VocabSize != es.Token.VocabSize {
		t.Error("VocabSize mismatch after JSON round-trip")
	}
}

func TestEmbeddingStack_InterfaceSizes(t *testing.T) {
	es := newTestStack(t)
	if es.InputSize() != es.Token.SeqLen {
		t.Errorf("InputSize %d, want %d", es.InputSize(), es.Token.SeqLen)
	}
	if es.OutputSize() != es.Token.SeqLen*es.Token.Dmodel {
		t.Errorf("OutputSize %d, want %d", es.OutputSize(), es.Token.SeqLen*es.Token.Dmodel)
	}
}

// ── sparse FD gradient check on TokenEmbedding ──────────────────────────────

func TestTokenEmbedding_SparseFDCheck(t *testing.T) {
	// Small vocab for tractable FD.
	te := NewTokenEmbedding[float64](5, 3, 4)
	rng := rand.New(rand.NewPCG(42, 0))
	te.Init(rng)

	ids := []int{1, 3, 1} // row 1 touched twice
	upstream := make([]float64, te.OutputSize())
	for i := range upstream {
		upstream[i] = rng.Float64() - 0.5
	}

	// Analytic gradient via Backward.
	if _, err := te.ForwardIDs(ids); err != nil {
		t.Fatal(err)
	}
	te.Backward(upstream)

	// FD per touched row, per dimension.
	const eps = 1e-5
	maxErr := 0.0
	for _, id := range []int{1, 3} {
		for d := range te.Dmodel {
			orig := te.Table[id*te.Dmodel+d]

			te.Table[id*te.Dmodel+d] = orig + eps
			if _, err := te.ForwardIDs(ids); err != nil {
				t.Fatal(err)
			}
			outp := te.lastOut
			var fp float64
			for i, u := range upstream {
				fp += u * outp[i]
			}

			te.Table[id*te.Dmodel+d] = orig - eps
			if _, err := te.ForwardIDs(ids); err != nil {
				t.Fatal(err)
			}
			outm := te.lastOut
			var fm float64
			for i, u := range upstream {
				fm += u * outm[i]
			}

			te.Table[id*te.Dmodel+d] = orig
			fd := (fp - fm) / (2 * eps)
			analytic := te.gradTable.Buf[id*te.Dmodel+d]
			diff := math.Abs(analytic - fd)
			if diff > maxErr {
				maxErr = diff
			}
		}
	}
	if maxErr > 1e-4 {
		t.Errorf("sparse FD check failed: max_abs_err = %e", maxErr)
	}
}
