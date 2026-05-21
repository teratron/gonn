package transformer

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"testing"
)

func newTestDecoder(seqLen, dmodel, numHeads, dff int) *DecoderBlock[float64] {
	cfg := newTestCfg(seqLen, dmodel, numHeads, dff)
	blk := NewDecoderBlock(cfg)
	blk.Init(rand.New(rand.NewPCG(7, 7)))
	return blk
}

// TestDecoderBlock_Forward verifies shape preservation (TRANS-1) and finite
// output on random initialization.
func TestDecoderBlock_Forward(t *testing.T) {
	seqLen, dmodel, numHeads, dff := 4, 8, 2, 16
	blk := newTestDecoder(seqLen, dmodel, numHeads, dff)

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.1
	}
	out := blk.Forward(x)

	if len(out) != seqLen*dmodel {
		t.Fatalf("Forward output len=%d, want %d", len(out), seqLen*dmodel)
	}
	for i, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("out[%d]=%v (NaN or Inf)", i, v)
		}
	}
}

// TestDecoderBlock_InputOutputSizes verifies InputSize and OutputSize.
func TestDecoderBlock_InputOutputSizes(t *testing.T) {
	seqLen, dmodel := 6, 12
	blk := newTestDecoder(seqLen, dmodel, 3, 24)
	want := seqLen * dmodel
	if blk.InputSize() != want {
		t.Errorf("InputSize=%d want %d", blk.InputSize(), want)
	}
	if blk.OutputSize() != want {
		t.Errorf("OutputSize=%d want %d", blk.OutputSize(), want)
	}
}

// TestDecoderBlock_CausalMaskPropagation verifies that the inner MHA is built
// with Causal=true so autoregressive masking is active (TRANS-4 / TRANS-C2).
func TestDecoderBlock_CausalMaskPropagation(t *testing.T) {
	blk := newTestDecoder(4, 8, 2, 16)
	if !blk.Attn.Causal {
		t.Error("DecoderBlock inner MHA: Causal=false, want true")
	}
}

// TestDecoderBlock_EncoderNotCausal verifies that EncoderBlock's MHA is NOT
// causal (TRANS-C2 negation — encoder uses full attention).
func TestDecoderBlock_EncoderNotCausal(t *testing.T) {
	enc := newTestEncoder(4, 8, 2, 16)
	if enc.Attn.Causal {
		t.Error("EncoderBlock inner MHA: Causal=true, want false")
	}
}

// TestDecoderBlock_SetPaddingMask verifies SetPaddingMask does not panic.
func TestDecoderBlock_SetPaddingMask(t *testing.T) {
	blk := newTestDecoder(4, 8, 2, 16)
	mask := make([]bool, 4)
	blk.SetPaddingMask(mask) // should not panic
}

// TestDecoderBlock_BackwardFD verifies ∂L/∂x via finite differences (TRANS-8).
// Tolerance 1e-3 to accommodate numerical noise from the causal-masked softmax.
func TestDecoderBlock_BackwardFD(t *testing.T) {
	const (
		seqLen   = 4
		dmodel   = 8
		numHeads = 2
		dff      = 16
		h        = 1e-4
		tol      = 1e-3
	)

	blk := newTestDecoder(seqLen, dmodel, numHeads, dff)
	blk.SetTraining(false)

	n := seqLen * dmodel
	x := make([]float64, n)
	for i := range x {
		x[i] = (float64(i) - float64(n)/2) * 0.05
	}
	upstream := make([]float64, n)
	for i := range upstream {
		upstream[i] = 1.0
	}

	blk.Forward(x)
	dxAnalytic := blk.Backward(upstream)

	loss := func(xIn []float64) float64 {
		out := blk.Forward(xIn)
		var s float64
		for i, u := range upstream {
			s += u * out[i]
		}
		return s
	}

	for i := range x {
		xp := make([]float64, n)
		xm := make([]float64, n)
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

// TestDecoderBlock_ApplyGradSGD verifies that ApplyGradSGD changes weights and
// produces different output on the next Forward call.
func TestDecoderBlock_ApplyGradSGD(t *testing.T) {
	blk := newTestDecoder(4, 8, 2, 16)
	blk.SetTraining(false)

	x := make([]float64, 4*8)
	for i := range x {
		x[i] = float64(i+1) * 0.1
	}
	upstream := make([]float64, 4*8)
	for i := range upstream {
		upstream[i] = 1.0
	}

	out1 := make([]float64, len(x))
	copy(out1, blk.Forward(x))
	blk.Backward(upstream)
	blk.ApplyGradSGD(0.1)
	out2 := blk.Forward(x)

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
	blk.Forward(x)
	blk.Backward(upstream)
	blk.ApplyGradSGD(0)
}

// TestDecoderBlock_JSON verifies MarshalJSON/UnmarshalJSON round-trip per TRANS-9.
func TestDecoderBlock_JSON(t *testing.T) {
	blk := newTestDecoder(4, 8, 2, 16)
	blk.SetTraining(false)

	x := make([]float64, 4*8)
	for i := range x {
		x[i] = float64(i+1) * 0.05
	}
	wantOut := blk.Forward(x)

	data, err := json.Marshal(blk)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}

	blk2 := &DecoderBlock[float64]{}
	if err := json.Unmarshal(data, blk2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	// Config round-trip
	if blk2.Cfg.SeqLen != blk.Cfg.SeqLen || blk2.Cfg.Dmodel != blk.Cfg.Dmodel ||
		blk2.Cfg.NumHeads != blk.Cfg.NumHeads || blk2.Cfg.Dff != blk.Cfg.Dff {
		t.Errorf("Config mismatch: got %+v want %+v", blk2.Cfg, blk.Cfg)
	}

	// Causal flag preserved through round-trip
	if !blk2.Attn.Causal {
		t.Error("UnmarshalJSON: Causal flag not restored")
	}

	// Forward output must match bit-exact after restore
	gotOut := blk2.Forward(x)
	if len(gotOut) != len(wantOut) {
		t.Fatalf("output len mismatch: got %d want %d", len(gotOut), len(wantOut))
	}
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Errorf("out[%d]: got %v want %v (not bit-exact after round-trip)", i, gotOut[i], wantOut[i])
		}
	}

	// Wrong Type tag must return error
	bad := []byte(`{"Type":"transformer.EncoderBlock","Config":{},"Attn":{},"Norm1":{},"Norm2":{},"FFN":{}}`)
	if err := json.Unmarshal(bad, &DecoderBlock[float64]{}); err == nil {
		t.Error("UnmarshalJSON: expected error for wrong Type tag, got nil")
	}
}

// TestDecoderBlock_PreNorm_Shape verifies shape preservation and finite output
// under the pre-norm wiring (TRANS-3).
func TestDecoderBlock_PreNorm_Shape(t *testing.T) {
	seqLen, dmodel, numHeads, dff := 4, 8, 2, 16
	cfg := newTestCfg(seqLen, dmodel, numHeads, dff)
	cfg.PreNorm = true
	blk := NewDecoderBlock(cfg)
	blk.Init(rand.New(rand.NewPCG(7, 7)))
	blk.SetTraining(false)

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.1
	}
	out := blk.Forward(x)

	if len(out) != seqLen*dmodel {
		t.Fatalf("PreNorm Forward len=%d want %d", len(out), seqLen*dmodel)
	}
	for i, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("out[%d]=%v (NaN or Inf)", i, v)
		}
	}
}

// TestDecoderBlock_DefaultActivationReLU verifies that Activation=0 is
// substituted to ReLU (TRANS-C4).
func TestDecoderBlock_DefaultActivationReLU(t *testing.T) {
	cfg := TransformerConfig[float64]{SeqLen: 2, Dmodel: 4, NumHeads: 1, Dff: 8}
	blk := NewDecoderBlock(cfg)
	if blk.Cfg.Activation != defaultActivation {
		t.Errorf("default Activation=%v, want ReLU", blk.Cfg.Activation)
	}
}

// TestDecoderBlock_DefaultDff verifies that Dff=0 is substituted to 4*Dmodel.
func TestDecoderBlock_DefaultDff(t *testing.T) {
	cfg := TransformerConfig[float64]{SeqLen: 2, Dmodel: 8, NumHeads: 2}
	blk := NewDecoderBlock(cfg)
	if blk.Cfg.Dff != 32 {
		t.Errorf("default Dff=%d, want 32 (4*8)", blk.Cfg.Dff)
	}
}
