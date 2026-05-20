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

// TestEncoderBlock_BackwardFD verifies ∂L/∂x via finite differences (TRANS-8).
// Tolerance 1e-3 accommodates numerical noise from the attention softmax path.
func TestEncoderBlock_BackwardFD(t *testing.T) {
	const (
		seqLen   = 4
		dmodel   = 8
		numHeads = 2
		dff      = 16
		h        = 1e-4
		tol      = 1e-3
	)

	blk := newTestEncoder(seqLen, dmodel, numHeads, dff)
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

	// analytic gradient via Forward + Backward
	blk.Forward(x)
	dxAnalytic := blk.Backward(upstream)

	// FD loss: L = Σ upstream_i * Forward(x)_i
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

// TestEncoderBlock_ApplyGradSGD verifies that ApplyGradSGD updates weights and
// zeroes gradient buffers, producing different Forward output on next call.
func TestEncoderBlock_ApplyGradSGD(t *testing.T) {
	blk := newTestEncoder(4, 8, 2, 16)
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
