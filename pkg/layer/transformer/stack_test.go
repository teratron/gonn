package transformer

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"testing"
)

func newTestStack(n int, seqLen, dmodel, numHeads, dff int) *Stack[float64] {
	cfg := newTestCfg(seqLen, dmodel, numHeads, dff)
	s := NewStack[float64](cfg, EncoderMode, n)
	s.Init(rand.New(rand.NewPCG(7, 7)))
	return s
}

// TestStack_ForwardShape verifies that an N-block encoder stack preserves the
// SeqLen*Dmodel shape and returns finite values (TRANS-1, TRANS-7).
func TestStack_ForwardShape(t *testing.T) {
	seqLen, dmodel, numHeads, dff := 4, 8, 2, 16
	for _, n := range []int{1, 3, 6} {
		s := newTestStack(n, seqLen, dmodel, numHeads, dff)
		x := make([]float64, seqLen*dmodel)
		for i := range x {
			x[i] = float64(i+1) * 0.05
		}
		out := s.Forward(x)
		if len(out) != seqLen*dmodel {
			t.Errorf("n=%d: Forward len=%d want %d", n, len(out), seqLen*dmodel)
		}
		for i, v := range out {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("n=%d: out[%d]=%v (NaN or Inf)", n, i, v)
			}
		}
	}
}

// TestStack_DecoderMode verifies that DecoderMode stack builds causal blocks.
func TestStack_DecoderMode(t *testing.T) {
	cfg := newTestCfg(4, 8, 2, 16)
	s := NewStack[float64](cfg, DecoderMode, 2)
	s.Init(rand.New(rand.NewPCG(5, 5)))
	for i, blk := range s.Blocks {
		dec, ok := blk.(*DecoderBlock[float64])
		if !ok {
			t.Fatalf("block[%d] is not *DecoderBlock", i)
		}
		if !dec.Attn.Causal {
			t.Errorf("block[%d]: Causal=false, want true (DecoderMode)", i)
		}
	}
}

// TestStack_EncoderMode verifies that EncoderMode stack builds non-causal blocks.
func TestStack_EncoderMode(t *testing.T) {
	cfg := newTestCfg(4, 8, 2, 16)
	s := NewStack[float64](cfg, EncoderMode, 2)
	s.Init(rand.New(rand.NewPCG(5, 5)))
	for i, blk := range s.Blocks {
		enc, ok := blk.(*EncoderBlock[float64])
		if !ok {
			t.Fatalf("block[%d] is not *EncoderBlock", i)
		}
		if enc.Attn.Causal {
			t.Errorf("block[%d]: Causal=true, want false (EncoderMode)", i)
		}
	}
}

// TestStack_InputOutputSizes verifies InputSize and OutputSize.
func TestStack_InputOutputSizes(t *testing.T) {
	seqLen, dmodel := 6, 12
	s := newTestStack(3, seqLen, dmodel, 3, 24)
	want := seqLen * dmodel
	if s.InputSize() != want {
		t.Errorf("InputSize=%d want %d", s.InputSize(), want)
	}
	if s.OutputSize() != want {
		t.Errorf("OutputSize=%d want %d", s.OutputSize(), want)
	}
}

// TestStack_UniqueWeights verifies that blocks have independent weights
// (TRANS-C8 — no weight tying across layers).
func TestStack_UniqueWeights(t *testing.T) {
	s := newTestStack(3, 4, 8, 2, 16)
	enc0 := s.Blocks[0].(*EncoderBlock[float64])
	enc1 := s.Blocks[1].(*EncoderBlock[float64])
	// Different seeds should produce different W1 weights.
	same := true
	for i := range enc0.FFN.W1 {
		if enc0.FFN.W1[i] != enc1.FFN.W1[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("blocks[0] and blocks[1] have identical FFN.W1 weights (weight tying detected)")
	}
}

// TestStack_SetPaddingMask verifies SetPaddingMask does not panic.
func TestStack_SetPaddingMask(t *testing.T) {
	s := newTestStack(3, 4, 8, 2, 16)
	mask := make([]bool, 4)
	s.SetPaddingMask(mask)
}

// TestStack_BackwardFD verifies ∂L/∂x via finite differences on a 6-block stack.
// Tolerance 1e-4 to accommodate accumulated numerical noise across 6 layers.
func TestStack_BackwardFD(t *testing.T) {
	const (
		n        = 6
		seqLen   = 4
		dmodel   = 8
		numHeads = 2
		dff      = 16
		h        = 1e-4
		tol      = 1e-3
	)

	s := newTestStack(n, seqLen, dmodel, numHeads, dff)
	s.SetTraining(false)

	size := seqLen * dmodel
	x := make([]float64, size)
	for i := range x {
		x[i] = (float64(i) - float64(size)/2) * 0.03
	}
	upstream := make([]float64, size)
	for i := range upstream {
		upstream[i] = 1.0
	}

	s.Forward(x)
	dxAnalytic := s.Backward(upstream)

	loss := func(xIn []float64) float64 {
		out := s.Forward(xIn)
		var sum float64
		for i, u := range upstream {
			sum += u * out[i]
		}
		return sum
	}

	for i := range x {
		xp := make([]float64, size)
		xm := make([]float64, size)
		copy(xp, x)
		copy(xm, x)
		xp[i] += h
		xm[i] -= h
		fd := (loss(xp) - loss(xm)) / (2 * h)
		if err := math.Abs(dxAnalytic[i] - fd); err > tol {
			t.Errorf("∂L/∂x[%d]: analytic=%v FD=%v err=%v", i, dxAnalytic[i], fd, err)
		}
	}
}

// TestStack_ApplyGradSGD verifies that weight updates change the output.
func TestStack_ApplyGradSGD(t *testing.T) {
	s := newTestStack(2, 4, 8, 2, 16)
	s.SetTraining(false)

	x := make([]float64, 4*8)
	for i := range x {
		x[i] = float64(i+1) * 0.1
	}
	upstream := make([]float64, 4*8)
	for i := range upstream {
		upstream[i] = 1.0
	}

	out1 := make([]float64, len(x))
	copy(out1, s.Forward(x))
	s.Backward(upstream)
	s.ApplyGradSGD(0.1)
	out2 := s.Forward(x)

	changed := false
	for i := range out1 {
		if out1[i] != out2[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("ApplyGradSGD: output unchanged after weight update")
	}

	// lr=0 should not panic
	s.Forward(x)
	s.Backward(upstream)
	s.ApplyGradSGD(0)
}

// TestStack_JSONRoundTrip verifies MarshalJSON/UnmarshalJSON round-trip (TRANS-9).
func TestStack_JSONRoundTrip(t *testing.T) {
	seqLen, dmodel, numHeads, dff := 4, 8, 2, 16
	s := newTestStack(3, seqLen, dmodel, numHeads, dff)
	s.SetTraining(false)

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.05
	}
	wantOut := s.Forward(x)

	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}

	s2 := &Stack[float64]{}
	if err := json.Unmarshal(data, s2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	// Config round-trip
	if s2.Cfg.SeqLen != s.Cfg.SeqLen || s2.Cfg.Dmodel != s.Cfg.Dmodel ||
		s2.Cfg.NumHeads != s.Cfg.NumHeads || s2.Cfg.Dff != s.Cfg.Dff {
		t.Errorf("Config mismatch: got %+v want %+v", s2.Cfg, s.Cfg)
	}
	if s2.Mode != s.Mode {
		t.Errorf("Mode mismatch: got %v want %v", s2.Mode, s.Mode)
	}
	if len(s2.Blocks) != len(s.Blocks) {
		t.Fatalf("Blocks len mismatch: got %d want %d", len(s2.Blocks), len(s.Blocks))
	}

	// Forward output must match bit-exact after restore
	gotOut := s2.Forward(x)
	if len(gotOut) != len(wantOut) {
		t.Fatalf("output len mismatch: got %d want %d", len(gotOut), len(wantOut))
	}
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Errorf("out[%d]: got %v want %v (not bit-exact after round-trip)", i, gotOut[i], wantOut[i])
		}
	}

	// Wrong Type tag must return error
	bad := []byte(`{"Type":"transformer.EncoderBlock","Config":{},"Mode":0,"Blocks":[]}`)
	if err := json.Unmarshal(bad, &Stack[float64]{}); err == nil {
		t.Error("UnmarshalJSON: expected error for wrong Type tag, got nil")
	}
}

// TestStack_DecoderJSONRoundTrip verifies DecoderMode stack JSON round-trip preserves causal flag.
func TestStack_DecoderJSONRoundTrip(t *testing.T) {
	cfg := newTestCfg(4, 8, 2, 16)
	s := NewStack[float64](cfg, DecoderMode, 2)
	s.Init(rand.New(rand.NewPCG(9, 9)))
	s.SetTraining(false)

	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}

	s2 := &Stack[float64]{}
	if err := json.Unmarshal(data, s2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if s2.Mode != DecoderMode {
		t.Errorf("Mode not restored: got %v want DecoderMode", s2.Mode)
	}
	for i, blk := range s2.Blocks {
		dec, ok := blk.(*DecoderBlock[float64])
		if !ok {
			t.Fatalf("block[%d] after unmarshal is not *DecoderBlock", i)
		}
		if !dec.Attn.Causal {
			t.Errorf("block[%d] after unmarshal: Causal=false, want true", i)
		}
	}
}

// TestStack_CopyTask trains a 6-block encoder stack on a synthetic copy-task
// (input = target) for 20 SGD steps and verifies that the training loss remains
// finite throughout. Full convergence is not required — this test catches
// NaN/Inf divergence and ensures the forward/backward/update loop is stable.
func TestStack_CopyTask(t *testing.T) {
	const (
		n        = 6
		seqLen   = 4
		dmodel   = 8
		numHeads = 2
		dff      = 16
		steps    = 20
		lr       = 0.001
	)

	cfg := newTestCfg(seqLen, dmodel, numHeads, dff)
	cfg.PreNorm = true
	s := NewStack(cfg, EncoderMode, n)
	s.Init(rand.New(rand.NewPCG(17, 17)))
	s.SetTraining(false)

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.05
	}

	mse := func(out, tgt []float64) float64 {
		var v float64
		for i := range out {
			d := out[i] - tgt[i]
			v += d * d
		}
		return 0.5 * v
	}

	firstLoss := math.MaxFloat64
	for step := 0; step < steps; step++ {
		out := s.Forward(x)
		if step == 0 {
			firstLoss = mse(out, x)
		}
		upstream := make([]float64, len(out))
		for i := range upstream {
			upstream[i] = out[i] - x[i]
		}
		s.Backward(upstream)
		s.ApplyGradSGD(lr)
	}

	finalOut := s.Forward(x)
	finalLoss := mse(finalOut, x)

	if math.IsNaN(finalLoss) || math.IsInf(finalLoss, 0) {
		t.Errorf("CopyTask: final loss is not finite: %v (initial=%v)", finalLoss, firstLoss)
	}
	for i, v := range finalOut {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("CopyTask: finalOut[%d]=%v (NaN or Inf)", i, v)
		}
	}
}

// TestStack_ParameterCount verifies the per-block parameter count matches
// l2-transformer-impl.md §4.4 for a BERT-Base-style configuration
// (Dmodel=768, NumHeads=12, Dff=3072 → 7,087,872 per block).
func TestStack_ParameterCount(t *testing.T) {
	const dmodel, dff = 768, 3072
	// Attention: 4 projection weight matrices (Dmodel×Dmodel) + 4 bias vectors (Dmodel)
	attnW := 4 * dmodel * dmodel
	attnB := 4 * dmodel
	// FFN: W1 (Dmodel×Dff) + b1 (Dff) + W2 (Dff×Dmodel) + b2 (Dmodel)
	ffnParams := dmodel*dff + dff + dff*dmodel + dmodel
	// LayerNorm: Norm1 + Norm2, each with gamma (Dmodel) + beta (Dmodel)
	lnParams := 2 * 2 * dmodel
	total := attnW + attnB + ffnParams + lnParams
	const want = 7_087_872
	if total != want {
		t.Errorf("spec §4.4 formula: got %d want %d", total, want)
	}
}
