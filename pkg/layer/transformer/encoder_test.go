package transformer

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
)

func newTestCfg(seqLen, dmodel, numHeads, dff int) TransformerConfig[float64] {
	return TransformerConfig[float64]{
		SeqLen:     seqLen,
		Dmodel:     dmodel,
		NumHeads:   numHeads,
		Dff:        dff,
		Activation: activation.ReLU,
	}
}

func newTestEncoder(seqLen, dmodel, numHeads, dff int) *EncoderBlock[float64] {
	cfg := newTestCfg(seqLen, dmodel, numHeads, dff)
	blk := NewEncoderBlock[float64](cfg)
	blk.Init(rand.New(rand.NewPCG(7, 7)))
	return blk
}

// TestEncoderBlock_Forward verifies shape preservation (TRANS-1) and finite
// output on random initialization.
func TestEncoderBlock_Forward(t *testing.T) {
	seqLen, dmodel, numHeads, dff := 4, 8, 2, 16
	blk := newTestEncoder(seqLen, dmodel, numHeads, dff)

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.1
	}

	out := blk.Forward(x)

	// TRANS-1: shape preservation
	if len(out) != seqLen*dmodel {
		t.Fatalf("Forward output len=%d, want %d", len(out), seqLen*dmodel)
	}

	// Check all outputs are finite (no NaN/Inf)
	for i, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("out[%d]=%v (NaN or Inf)", i, v)
		}
	}
}

// TestEncoderBlock_ForwardDifferentInputs verifies that different inputs
// produce different outputs (sanity check that the block is not trivial).
func TestEncoderBlock_ForwardDifferentInputs(t *testing.T) {
	seqLen, dmodel, numHeads, dff := 4, 8, 2, 16
	blk := newTestEncoder(seqLen, dmodel, numHeads, dff)

	x1 := make([]float64, seqLen*dmodel)
	x2 := make([]float64, seqLen*dmodel)
	for i := range x1 {
		x1[i] = float64(i) * 0.1
		x2[i] = float64(i)*0.1 + 1.0
	}

	out1 := blk.Forward(x1)
	blk2 := newTestEncoder(seqLen, dmodel, numHeads, dff)
	out2 := blk2.Forward(x2)

	var same bool = true
	for i := range out1 {
		if out1[i] != out2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different inputs produced identical outputs")
	}
}

// TestEncoderBlock_InputOutputSizes verifies InputSize and OutputSize.
func TestEncoderBlock_InputOutputSizes(t *testing.T) {
	seqLen, dmodel := 6, 12
	blk := newTestEncoder(seqLen, dmodel, 3, 24)
	want := seqLen * dmodel
	if blk.InputSize() != want {
		t.Errorf("InputSize=%d want %d", blk.InputSize(), want)
	}
	if blk.OutputSize() != want {
		t.Errorf("OutputSize=%d want %d", blk.OutputSize(), want)
	}
}

// TestEncoderBlock_SetPaddingMask verifies that SetPaddingMask is a no-panic
// forwarding call.
func TestEncoderBlock_SetPaddingMask(t *testing.T) {
	blk := newTestEncoder(4, 8, 2, 16)
	mask := make([]bool, 4)
	blk.SetPaddingMask(mask) // should not panic
}

// TestEncoderBlock_DefaultActivationReLU verifies that Activation=0 is
// substituted to ReLU per TRANS-C4.
func TestEncoderBlock_DefaultActivationReLU(t *testing.T) {
	cfg := TransformerConfig[float64]{SeqLen: 2, Dmodel: 4, NumHeads: 1, Dff: 8}
	// Activation is intentionally left zero.
	blk := NewEncoderBlock[float64](cfg)
	if blk.Cfg.Activation != activation.ReLU {
		t.Errorf("default Activation=%v, want ReLU", blk.Cfg.Activation)
	}
}

// TestEncoderBlock_DefaultDff verifies that Dff=0 is substituted to 4*Dmodel.
func TestEncoderBlock_DefaultDff(t *testing.T) {
	cfg := TransformerConfig[float64]{SeqLen: 2, Dmodel: 8, NumHeads: 2}
	blk := NewEncoderBlock[float64](cfg)
	if blk.Cfg.Dff != 32 {
		t.Errorf("default Dff=%d, want 32 (4*8)", blk.Cfg.Dff)
	}
}
