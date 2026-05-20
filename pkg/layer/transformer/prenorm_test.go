package transformer

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestPreNorm_Shape verifies that pre-norm (TRANS-3) preserves SeqLen*Dmodel
// shape identically to post-norm.
func TestPreNorm_Shape(t *testing.T) {
	const seqLen, dmodel, numHeads, dff = 4, 8, 2, 16
	cfgPost := newTestCfg(seqLen, dmodel, numHeads, dff)
	cfgPre := cfgPost
	cfgPre.PreNorm = true

	rng := rand.New(rand.NewPCG(11, 11))
	enc := newTestBlock(cfgPost, rng)
	rng = rand.New(rand.NewPCG(11, 11))
	encPre := newTestBlock(cfgPre, rng)

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.05
	}
	outPost := enc.Forward(x)
	outPre := encPre.Forward(x)

	want := seqLen * dmodel
	if len(outPost) != want || len(outPre) != want {
		t.Errorf("shape mismatch: post=%d pre=%d want=%d", len(outPost), len(outPre), want)
	}
}

// TestPreNorm_BackwardFD validates ∂L/∂x for pre-norm EncoderBlock via
// finite differences (TRANS-3 — same grad check as post-norm, tol=1e-3).
func TestPreNorm_BackwardFD(t *testing.T) {
	const (
		seqLen   = 4
		dmodel   = 8
		numHeads = 2
		dff      = 16
		h        = 1e-4
		tol      = 1e-3
	)

	cfg := newTestCfg(seqLen, dmodel, numHeads, dff)
	cfg.PreNorm = true
	blk := newTestBlock(cfg, rand.New(rand.NewPCG(13, 13)))
	blk.SetTraining(false)

	size := seqLen * dmodel
	x := make([]float64, size)
	for i := range x {
		x[i] = (float64(i) - float64(size)/2) * 0.04
	}
	upstream := make([]float64, size)
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
		xp := make([]float64, size)
		xm := make([]float64, size)
		copy(xp, x)
		copy(xm, x)
		xp[i] += h
		xm[i] -= h
		fd := (loss(xp) - loss(xm)) / (2 * h)
		if err := math.Abs(dxAnalytic[i] - fd); err > tol {
			t.Errorf("pre-norm ∂L/∂x[%d]: analytic=%v FD=%v err=%v", i, dxAnalytic[i], fd, err)
		}
	}
}

// TestPreNorm_GradientFlowDepth12 verifies that a 12-block pre-norm Stack has
// non-vanishing ∂L/∂x (gradient-flow sanity at depth 12, TRANS-3 §spec).
// Passes when at least one input gradient component has |dx| > 1e-8.
func TestPreNorm_GradientFlowDepth12(t *testing.T) {
	const n, seqLen, dmodel, numHeads, dff = 12, 4, 8, 2, 16
	cfg := newTestCfg(seqLen, dmodel, numHeads, dff)
	cfg.PreNorm = true
	s := NewStack(cfg, EncoderMode, n)
	s.Init(rand.New(rand.NewPCG(7, 7)))
	s.SetTraining(false)

	size := seqLen * dmodel
	x := make([]float64, size)
	for i := range x {
		x[i] = float64(i+1) * 0.05
	}
	upstream := make([]float64, size)
	for i := range upstream {
		upstream[i] = 1.0
	}

	s.Forward(x)
	dx := s.Backward(upstream)

	anyNonZero := false
	for _, v := range dx {
		if math.Abs(v) > 1e-8 {
			anyNonZero = true
			break
		}
	}
	if !anyNonZero {
		t.Error("depth-12 pre-norm stack: all input gradients ≈ 0 (gradient vanished)")
	}

	for i, v := range dx {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("dx[%d]=%v (NaN or Inf)", i, v)
		}
	}
}

// TestPreNormVsPostNorm_OutputsDiffer asserts that pre-norm and post-norm
// produce DIFFERENT outputs for the same initialized weights — confirming the
// two wiring paths are distinct (they should not be accidentally equal).
func TestPreNormVsPostNorm_OutputsDiffer(t *testing.T) {
	const seqLen, dmodel, numHeads, dff = 4, 8, 2, 16

	cfgPost := newTestCfg(seqLen, dmodel, numHeads, dff)
	cfgPre := cfgPost
	cfgPre.PreNorm = true

	seed := rand.NewPCG(99, 99)
	blkPost := newTestBlock(cfgPost, rand.New(seed))
	seed2 := rand.NewPCG(99, 99)
	blkPre := newTestBlock(cfgPre, rand.New(seed2))

	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = float64(i+1) * 0.1
	}

	outPost := blkPost.Forward(x)
	outPre := blkPre.Forward(x)

	same := true
	for i := range outPost {
		if outPost[i] != outPre[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("pre-norm and post-norm produced identical outputs — wiring error suspected")
	}
}

// newTestBlock is a helper that creates and Init-s an EncoderBlock[float64].
func newTestBlock(cfg TransformerConfig[float64], rng *rand.Rand) *EncoderBlock[float64] {
	blk := NewEncoderBlock(cfg)
	blk.Init(rng)
	return blk
}
