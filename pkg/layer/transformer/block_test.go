package transformer

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
)

func TestAddInPlace(t *testing.T) {
	dst := []float64{1, 2, 3, 4}
	src := []float64{10, 20, 30, 40}
	addInPlace(dst, src)
	want := []float64{11, 22, 33, 44}
	for i, v := range dst {
		if v != want[i] {
			t.Errorf("[%d] got %v want %v", i, v, want[i])
		}
	}
}

func TestAddInPlace_Zero(t *testing.T) {
	dst := []float64{5, -3, 0}
	src := []float64{0, 0, 0}
	addInPlace(dst, src)
	if dst[0] != 5 || dst[1] != -3 || dst[2] != 0 {
		t.Errorf("addInPlace with zero src changed dst: %v", dst)
	}
}

func newTestFFN(dmodel, dff int) *ffn[float64] {
	f := newFFN[float64](1, dmodel, dff, activation.ReLU)
	f.Init(rand.New(rand.NewPCG(42, 42)))
	return f
}

func TestFFN_ForwardShape(t *testing.T) {
	cases := []struct{ dmodel, dff int }{
		{4, 8},
		{8, 16},
		{16, 32},
	}
	for _, c := range cases {
		f := newTestFFN(c.dmodel, c.dff)
		x := make([]float64, c.dmodel)
		for i := range x {
			x[i] = float64(i + 1)
		}
		y := f.Forward(x)
		if len(y) != c.dmodel {
			t.Errorf("dmodel=%d dff=%d: Forward len=%d want %d", c.dmodel, c.dff, len(y), c.dmodel)
		}
	}
}

func TestFFN_ForwardFinite(t *testing.T) {
	f := newTestFFN(4, 8)
	x := []float64{1.0, -0.5, 0.3, 2.1}
	y := f.Forward(x)
	for i, v := range y {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("y[%d]=%v (NaN/Inf)", i, v)
		}
	}
}

// ffnLoss computes L = Σ upstream_i * FFN(x)_i for FD checks.
// Uses a fresh FFN with identical weights each call.
func ffnLoss(f *ffn[float64], x, upstream []float64) float64 {
	y := f.Forward(x)
	var s float64
	for i, u := range upstream {
		s += u * y[i]
	}
	return s
}

// cloneFFN returns a deep copy of f's weight tensors (not cache).
func cloneFFN(src *ffn[float64]) *ffn[float64] {
	dst := &ffn[float64]{
		W1:      make([]float64, len(src.W1)),
		b1:      make([]float64, len(src.b1)),
		W2:      make([]float64, len(src.W2)),
		b2:      make([]float64, len(src.b2)),
		gW1:     make([]float64, len(src.gW1)),
		gb1:     make([]float64, len(src.gb1)),
		gW2:     make([]float64, len(src.gW2)),
		gb2:     make([]float64, len(src.gb2)),
		seqLen:  src.seqLen,
		dmodel:  src.dmodel,
		dff:     src.dff,
		actMode: src.actMode,
	}
	copy(dst.W1, src.W1)
	copy(dst.b1, src.b1)
	copy(dst.W2, src.W2)
	copy(dst.b2, src.b2)
	return dst
}

// TestFFN_BackwardFD verifies all gradient groups via finite differences.
// Threshold: max_abs_err < 1e-4 for T=float64.
func TestFFN_BackwardFD(t *testing.T) {
	const (
		dmodel = 4
		dff    = 8
		h      = 1e-5
		tol    = 1e-4
	)

	base := newTestFFN(dmodel, dff)
	x := []float64{0.5, -1.2, 0.3, 2.1}
	upstream := []float64{1.0, -0.5, 0.3, 0.7}

	// Compute analytic gradients via forward + backward.
	base.Forward(x)
	dxAnalytic := base.Backward(upstream)

	// --- ∂L/∂x ---
	for i := range x {
		xp := make([]float64, dmodel)
		xm := make([]float64, dmodel)
		copy(xp, x)
		copy(xm, x)
		xp[i] += h
		xm[i] -= h
		fp := cloneFFN(base)
		fm := cloneFFN(base)
		lp := ffnLoss(fp, xp, upstream)
		lm := ffnLoss(fm, xm, upstream)
		fd := (lp - lm) / (2 * h)
		if err := math.Abs(dxAnalytic[i] - fd); err > tol {
			t.Errorf("∂L/∂x[%d]: analytic=%v FD=%v err=%v", i, dxAnalytic[i], fd, err)
		}
	}

	// --- ∂L/∂W1 ---
	for i := range base.W1 {
		fp := cloneFFN(base)
		fm := cloneFFN(base)
		fp.W1[i] += h
		fm.W1[i] -= h
		lp := ffnLoss(fp, x, upstream)
		lm := ffnLoss(fm, x, upstream)
		fd := (lp - lm) / (2 * h)
		if err := math.Abs(base.gW1[i] - fd); err > tol {
			t.Errorf("∂L/∂W1[%d]: analytic=%v FD=%v err=%v", i, base.gW1[i], fd, err)
		}
	}

	// --- ∂L/∂b1 ---
	for i := range base.b1 {
		fp := cloneFFN(base)
		fm := cloneFFN(base)
		fp.b1[i] += h
		fm.b1[i] -= h
		lp := ffnLoss(fp, x, upstream)
		lm := ffnLoss(fm, x, upstream)
		fd := (lp - lm) / (2 * h)
		if err := math.Abs(base.gb1[i] - fd); err > tol {
			t.Errorf("∂L/∂b1[%d]: analytic=%v FD=%v err=%v", i, base.gb1[i], fd, err)
		}
	}

	// --- ∂L/∂W2 ---
	for i := range base.W2 {
		fp := cloneFFN(base)
		fm := cloneFFN(base)
		fp.W2[i] += h
		fm.W2[i] -= h
		lp := ffnLoss(fp, x, upstream)
		lm := ffnLoss(fm, x, upstream)
		fd := (lp - lm) / (2 * h)
		if err := math.Abs(base.gW2[i] - fd); err > tol {
			t.Errorf("∂L/∂W2[%d]: analytic=%v FD=%v err=%v", i, base.gW2[i], fd, err)
		}
	}

	// --- ∂L/∂b2 ---
	for i := range base.b2 {
		fp := cloneFFN(base)
		fm := cloneFFN(base)
		fp.b2[i] += h
		fm.b2[i] -= h
		lp := ffnLoss(fp, x, upstream)
		lm := ffnLoss(fm, x, upstream)
		fd := (lp - lm) / (2 * h)
		if err := math.Abs(base.gb2[i] - fd); err > tol {
			t.Errorf("∂L/∂b2[%d]: analytic=%v FD=%v err=%v", i, base.gb2[i], fd, err)
		}
	}
}

// TestFFN_ApplyGradSGD verifies weight update and gradient zeroing.
func TestFFN_ApplyGradSGD(t *testing.T) {
	f := newTestFFN(4, 8)
	x := []float64{1, 2, 3, 4}
	upstream := []float64{0.1, -0.2, 0.3, 0.4}
	f.Forward(x)
	f.Backward(upstream)

	w1Before := make([]float64, len(f.W1))
	copy(w1Before, f.W1)
	gW1Before := make([]float64, len(f.gW1))
	copy(gW1Before, f.gW1)

	f.ApplyGradSGD(0.01)

	for i := range f.W1 {
		want := w1Before[i] - 0.01*gW1Before[i]
		if math.Abs(f.W1[i]-want) > 1e-12 {
			t.Errorf("W1[%d]: got %v want %v", i, f.W1[i], want)
		}
		if f.gW1[i] != 0 {
			t.Errorf("gW1[%d] not zeroed after ApplyGradSGD", i)
		}
	}
	for i := range f.gb1 {
		if f.gb1[i] != 0 {
			t.Errorf("gb1[%d] not zeroed", i)
		}
	}
	for i := range f.gW2 {
		if f.gW2[i] != 0 {
			t.Errorf("gW2[%d] not zeroed", i)
		}
	}
	for i := range f.gb2 {
		if f.gb2[i] != 0 {
			t.Errorf("gb2[%d] not zeroed", i)
		}
	}
}
