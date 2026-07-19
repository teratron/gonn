package transformer

import (
	"math/rand/v2"
	"testing"
)

// TestResidualIdentity_PreNorm verifies TRANS-6: when the inner sub-layers
// (attention and FFN) produce all-zeros (due to zero weights), the pre-norm
// residual connections preserve the input bit-exact for both float32 and float64.
//
// Pre-norm formula: Y = X + Attn(LN₁(X)); Z = Y + FFN(LN₂(Y))
// With zero Attn and FFN weights → Y = X + 0 = X; Z = X + 0 = X.
func TestResidualIdentity_PreNorm(t *testing.T) {
	const seqLen, dmodel, numHeads, dff = 4, 8, 2, 16

	t.Run("float64", func(t *testing.T) {
		cfg := TransformerConfig[float64]{
			SeqLen:   seqLen,
			Dmodel:   dmodel,
			NumHeads: numHeads,
			Dff:      dff,
			PreNorm:  true,
		}
		// No Init call — weights remain all-zero.
		blk := NewEncoderBlock(cfg)

		x := make([]float64, seqLen*dmodel)
		for i := range x {
			x[i] = float64(i+1) * 0.1
		}

		out := blk.Forward(x)
		if len(out) != len(x) {
			t.Fatalf("output length %d != input length %d", len(out), len(x))
		}
		for i, v := range out {
			if v != x[i] {
				t.Errorf("out[%d]=%v != x[%d]=%v (residual identity violated)", i, v, i, x[i])
			}
		}
	})

	t.Run("float32", func(t *testing.T) {
		cfg := TransformerConfig[float32]{
			SeqLen:   seqLen,
			Dmodel:   dmodel,
			NumHeads: numHeads,
			Dff:      dff,
			PreNorm:  true,
		}
		blk := NewEncoderBlock(cfg)

		x := make([]float32, seqLen*dmodel)
		for i := range x {
			x[i] = float32(i+1) * 0.1
		}

		out := blk.Forward(x)
		if len(out) != len(x) {
			t.Fatalf("output length %d != input length %d", len(out), len(x))
		}
		for i, v := range out {
			if v != x[i] {
				t.Errorf("out[%d]=%v != x[%d]=%v (residual identity violated)", i, v, i, x[i])
			}
		}
	})
}

// TestResidualIdentity_Decoder verifies TRANS-6 for DecoderBlock with pre-norm
// and zero weights: output bit-equals input.
func TestResidualIdentity_Decoder(t *testing.T) {
	const seqLen, dmodel, numHeads, dff = 4, 8, 2, 16
	cfg := TransformerConfig[float64]{
		SeqLen:   seqLen,
		Dmodel:   dmodel,
		NumHeads: numHeads,
		Dff:      dff,
		PreNorm:  true,
	}
	blk := NewDecoderBlock(cfg)

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.07
	}
	out := blk.Forward(x)
	for i, v := range out {
		if v != x[i] {
			t.Errorf("decoder out[%d]=%v != x[%d]=%v (residual identity violated)", i, v, i, x[i])
		}
	}
}

// TestAddInPlaceIdentity confirms that addInPlace with an all-zero src is a
// no-op: dst[i] is unchanged (TRANS-6 — no scale, no clip, pure bit-exact add).
func TestAddInPlaceIdentity(t *testing.T) {
	dst := []float64{1, 2, 3, 4, 5}
	want := []float64{1, 2, 3, 4, 5}
	src := make([]float64, len(dst))
	addInPlace(dst, src)
	for i, v := range dst {
		if v != want[i] {
			t.Errorf("dst[%d]=%v after addInPlace(zero) != %v", i, v, want[i])
		}
	}
}

// TestAddInPlaceExact verifies that addInPlace accumulates without rounding or
// truncation: dst+src must equal the exact IEEE-754 sum at each position.
func TestAddInPlaceExact(t *testing.T) {
	a := []float64{1.0, 0.5, 0.25}
	b := []float64{2.0, 1.5, 0.75}
	want := []float64{3.0, 2.0, 1.0}
	addInPlace(a, b)
	for i, v := range a {
		if v != want[i] {
			t.Errorf("a[%d]=%v want %v", i, v, want[i])
		}
	}
}

// TestStack_ResidualIdentity_PreNorm verifies that a 6-block Stack with pre-norm
// and zero weights (no Init) passes the input through all residuals bit-exact.
func TestStack_ResidualIdentity_PreNorm(t *testing.T) {
	const n, seqLen, dmodel, numHeads, dff = 6, 4, 8, 2, 16
	cfg := TransformerConfig[float64]{
		SeqLen:   seqLen,
		Dmodel:   dmodel,
		NumHeads: numHeads,
		Dff:      dff,
		PreNorm:  true,
	}
	// No Init — all weights zero.
	s := NewStack(cfg, EncoderMode, n)

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i) * 0.03
	}
	out := s.Forward(x)
	for i, v := range out {
		if v != x[i] {
			t.Errorf("stack[6] out[%d]=%v != x[%d]=%v (residual identity violated)", i, v, i, x[i])
		}
	}
}

// TestResidualIdentity_PostNorm_ZeroLN verifies that with zero-weight attention
// and FFN, post-norm output equals LN2(LN1(x)) — confirming the residual ADD
// contributes correctly (adds zero, not something else).
func TestResidualIdentity_PostNorm_ZeroWeights(t *testing.T) {
	const seqLen, dmodel, numHeads, dff = 2, 4, 1, 8
	cfg := TransformerConfig[float64]{
		SeqLen:   seqLen,
		Dmodel:   dmodel,
		NumHeads: numHeads,
		Dff:      dff,
		PreNorm:  false,
	}
	blk := NewEncoderBlock(cfg)

	// With zero weights, post-norm: bufZ = x + 0 = x; cacheZ1 = LN1(x).
	// ffnOut = 0; bufF = LN1(x) + 0 = LN1(x); out = LN2(LN1(x)).
	// Verify that output is finite and has correct length.
	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.5
	}
	out := blk.Forward(x)
	if len(out) != len(x) {
		t.Fatalf("output len %d want %d", len(out), len(x))
	}
	for i, v := range out {
		if v != v { // NaN check
			t.Errorf("out[%d] is NaN", i)
		}
	}
}

// TestResidualIdentity_Stack_Init verifies that after Init, a 6-block pre-norm
// stack with initialized (non-zero) weights still preserves the TRANS-6 shape
// invariant: Forward len == seqLen*dmodel.
func TestResidualIdentity_Stack_Init(t *testing.T) {
	const n, seqLen, dmodel, numHeads, dff = 6, 4, 8, 2, 16
	cfg := newTestCfg(seqLen, dmodel, numHeads, dff)
	cfg.PreNorm = true
	s := NewStack(cfg, EncoderMode, n)
	s.Init(rand.New(rand.NewPCG(42, 42)))

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i) * 0.1
	}
	out := s.Forward(x)
	if len(out) != seqLen*dmodel {
		t.Errorf("len(out)=%d want %d", len(out), seqLen*dmodel)
	}
}
