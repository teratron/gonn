package cpu

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// naiveMatVec / naiveMatVecT / naiveGradOuter are deliberately written from the
// mathematical definition with the "obvious" index arithmetic, independent of
// the shipped implementations' loop ordering and row-slicing. They exist so the
// kernels are checked against the spec rather than against themselves.
func naiveMatVec(w []float64, in, out int, input []float64) []float64 {
	preact := make([]float64, out)
	for o := range out {
		for j := range in {
			preact[o] += w[o*in+j] * input[j]
		}
	}
	return preact
}

func naiveMatVecT(w []float64, in, out int, delta []float64) []float64 {
	dIn := make([]float64, in)
	for j := range in {
		for o := range out {
			dIn[j] += delta[o] * w[o*in+j]
		}
	}
	return dIn
}

func naiveGradOuter(in, out int, delta, input []float64, scale float64) []float64 {
	g := make([]float64, in*out)
	for o := range out {
		for j := range in {
			g[o*in+j] = scale * delta[o] * input[j]
		}
	}
	return g
}

// TestDenseKernelsMatchDefinition sweeps a range of shapes — including the
// degenerate 1×1 and strongly rectangular cases — and checks all three kernels
// against the independent definitions.
func TestDenseKernelsMatchDefinition(t *testing.T) {
	b := Backend[float64]{}
	rng := rand.New(rand.NewPCG(11, 22))
	shapes := [][2]int{{1, 1}, {1, 7}, {7, 1}, {3, 4}, {4, 3}, {16, 9}, {9, 16}}

	for _, sh := range shapes {
		out, in := sh[0], sh[1]
		w := make([]float64, in*out)
		for i := range w {
			w[i] = rng.NormFloat64()
		}
		input := make([]float64, in)
		for i := range input {
			input[i] = rng.NormFloat64()
		}
		delta := make([]float64, out)
		for i := range delta {
			delta[i] = rng.NormFloat64()
		}
		m := compute.DenseMatrix[float64]{W: w, In: in, Out: out}

		preact := make([]float64, out)
		if err := b.MatVec(m, input, preact); err != nil {
			t.Fatalf("shape %dx%d MatVec: %v", out, in, err)
		}
		for i, want := range naiveMatVec(w, in, out, input) {
			if math.Abs(preact[i]-want) > 1e-12 {
				t.Errorf("shape %dx%d MatVec[%d] = %.15g, want %.15g", out, in, i, preact[i], want)
			}
		}

		dIn := make([]float64, in)
		if err := b.MatVecT(m, delta, dIn); err != nil {
			t.Fatalf("shape %dx%d MatVecT: %v", out, in, err)
		}
		for i, want := range naiveMatVecT(w, in, out, delta) {
			if math.Abs(dIn[i]-want) > 1e-12 {
				t.Errorf("shape %dx%d MatVecT[%d] = %.15g, want %.15g", out, in, i, dIn[i], want)
			}
		}

		grad := make([]float64, in*out)
		if err := b.GradOuter(m, delta, input, grad, -1); err != nil {
			t.Fatalf("shape %dx%d GradOuter: %v", out, in, err)
		}
		for i, want := range naiveGradOuter(in, out, delta, input, -1) {
			if math.Abs(grad[i]-want) > 1e-12 {
				t.Errorf("shape %dx%d GradOuter[%d] = %.15g, want %.15g", out, in, i, grad[i], want)
			}
		}
	}
}

// TestMatVecTOverwritesStaleResults guards the sparse-delta shortcut: MatVecT
// skips zero deltas, so it MUST clear dInput first or a zeroed delta would
// leave the previous call's value in place.
func TestMatVecTOverwritesStaleResults(t *testing.T) {
	b := Backend[float64]{}
	m := compute.DenseMatrix[float64]{W: []float64{1, 2, 3, 4}, In: 2, Out: 2}
	dIn := []float64{99, 99}
	if err := b.MatVecT(m, []float64{0, 0}, dIn); err != nil {
		t.Fatalf("MatVecT: %v", err)
	}
	for i, v := range dIn {
		if v != 0 {
			t.Errorf("dInput[%d] = %v, want 0 — stale result survived an all-zero delta", i, v)
		}
	}
}

// TestDenseKernelsRejectBadShapes: every shape violation must surface as
// ErrCompute rather than a panic or a silent partial write.
func TestDenseKernelsRejectBadShapes(t *testing.T) {
	b := Backend[float64]{}
	good := compute.DenseMatrix[float64]{W: []float64{1, 2, 3, 4, 5, 6}, In: 3, Out: 2}

	cases := []struct {
		name string
		call func() error
	}{
		{"matvec_short_input", func() error {
			return b.MatVec(good, make([]float64, 2), make([]float64, 2))
		}},
		{"matvec_short_preact", func() error {
			return b.MatVec(good, make([]float64, 3), make([]float64, 1))
		}},
		{"matvect_short_delta", func() error {
			return b.MatVecT(good, make([]float64, 1), make([]float64, 3))
		}},
		{"gradouter_short_grad", func() error {
			return b.GradOuter(good, make([]float64, 2), make([]float64, 3), make([]float64, 5), 1)
		}},
		{"ragged_weights", func() error {
			bad := compute.DenseMatrix[float64]{W: []float64{1, 2, 3}, In: 3, Out: 2}
			return b.MatVec(bad, make([]float64, 3), make([]float64, 2))
		}},
		{"degenerate_shape", func() error {
			bad := compute.DenseMatrix[float64]{W: nil, In: 0, Out: 0}
			return b.MatVec(bad, nil, nil)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.call()
			if err == nil {
				t.Fatal("expected an error, got nil")
			}
			if !errors.Is(err, utils.ErrCompute) {
				t.Errorf("error %v is not ErrCompute", err)
			}
		})
	}
}

// TestBackendImplementsDenseKernels pins the capability contract the engine
// type-asserts on: if this stops holding, every network silently reverts to the
// internal reference loops.
func TestBackendImplementsDenseKernels(t *testing.T) {
	var b any = Backend[float32]{}
	if _, ok := b.(compute.DenseKernels[float32]); !ok {
		t.Error("cpu.Backend[float32] no longer implements compute.DenseKernels")
	}
	var b64 any = Backend[float64]{}
	if _, ok := b64.(compute.DenseKernels[float64]); !ok {
		t.Error("cpu.Backend[float64] no longer implements compute.DenseKernels")
	}
}
