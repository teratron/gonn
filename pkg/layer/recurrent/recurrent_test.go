package recurrent

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
)

// TestCellHelpers verifies that applySigmoidFused and applyTanhFused match the
// activation-package functions element-wise within 1e-12.
func TestCellHelpers(t *testing.T) {
	t.Parallel()
	inputs := []float64{-3, -1, -0.5, 0, 0.5, 1, 3, 10, -10}

	t.Run("sigmoid", func(t *testing.T) {
		t.Parallel()
		gates := make([]float64, len(inputs))
		copy(gates, inputs)
		applySigmoidFused(gates, 0, len(gates))
		for i, v := range inputs {
			want := float64(activation.Activation(v, activation.SIGMOID))
			if math.Abs(gates[i]-want) > 1e-12 {
				t.Errorf("sigmoid[%d] got %.15g, want %.15g (diff %.3e)", i, gates[i], want, math.Abs(gates[i]-want))
			}
		}
	})

	t.Run("tanh", func(t *testing.T) {
		t.Parallel()
		gates := make([]float64, len(inputs))
		copy(gates, inputs)
		applyTanhFused(gates, 0, len(gates))
		for i, v := range inputs {
			want := float64(activation.Activation(v, activation.TanH))
			if math.Abs(gates[i]-want) > 1e-12 {
				t.Errorf("tanh[%d] got %.15g, want %.15g (diff %.3e)", i, gates[i], want, math.Abs(gates[i]-want))
			}
		}
	})

	t.Run("sigmoid_offset", func(t *testing.T) {
		t.Parallel()
		gates := make([]float64, 2*len(inputs))
		// Fill first half with sentinel value 42 (should be untouched).
		for i := range len(inputs) {
			gates[i] = 42
		}
		copy(gates[len(inputs):], inputs)
		applySigmoidFused(gates, len(inputs), len(inputs))
		for i := range len(inputs) {
			if gates[i] != 42 {
				t.Fatalf("offset: sentinel at index %d was modified", i)
			}
		}
		for i, v := range inputs {
			want := float64(activation.Activation(v, activation.SIGMOID))
			got := gates[len(inputs)+i]
			if math.Abs(got-want) > 1e-12 {
				t.Errorf("sigmoid offset[%d] got %.15g, want %.15g", i, got, want)
			}
		}
	})

	t.Run("matVecAdd", func(t *testing.T) {
		t.Parallel()
		// 2×3 matrix (row-major), multiplied by [1, 2, 3] → [14, 32].
		mat := []float64{1, 2, 3, 4, 5, 6}
		vec := []float64{1, 2, 3}
		dst := []float64{0, 0}
		matVecAdd(dst, mat, vec, 2, 3)
		if math.Abs(dst[0]-14) > 1e-12 || math.Abs(dst[1]-32) > 1e-12 {
			t.Errorf("matVecAdd = [%v, %v], want [14, 32]", dst[0], dst[1])
		}
		// Second call accumulates (dst += ...).
		matVecAdd(dst, mat, vec, 2, 3)
		if math.Abs(dst[0]-28) > 1e-12 || math.Abs(dst[1]-64) > 1e-12 {
			t.Errorf("matVecAdd accumulate = [%v, %v], want [28, 64]", dst[0], dst[1])
		}
	})

	t.Run("float32", func(t *testing.T) {
		t.Parallel()
		gates32 := make([]float32, len(inputs))
		for i, v := range inputs {
			gates32[i] = float32(v)
		}
		applySigmoidFused(gates32, 0, len(gates32))
		for i, v := range inputs {
			want := float32(activation.Activation(float32(v), activation.SIGMOID))
			if math.Abs(float64(gates32[i]-want)) > 1e-6 {
				t.Errorf("sigmoid float32[%d] got %v, want %v", i, gates32[i], want)
			}
		}
	})
}
