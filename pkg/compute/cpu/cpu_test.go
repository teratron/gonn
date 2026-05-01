// Package cpu — kernel correctness and registration tests.
//
// Covers COMP-1 (reference kernels match hand-computed values),
// COMP-2 (registration) and COMP-4 (Allocate/Free buffer lifecycle).
package cpu

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// makeXORWeights builds a tiny 2 → 2 layer hand-tuned for the XOR
// hidden-layer reference math. Activation is ReLU so the math stays
// auditable in the test body.
func makeXORWeights() compute.LayerHandle[float32] {
	return compute.LayerHandle[float32]{
		Size: 2,
		Weights: [][]float32{
			{1, 1},
			{1, 1},
		},
		Bias:       []float32{0, -1},
		Activation: relu[float32],
		Derivative: reluDeriv[float32],
	}
}

func relu[T utils.Float](v T) T {
	if v > 0 {
		return v
	}
	return 0
}

func reluDeriv[T utils.Float](v T) T {
	if v > 0 {
		return 1
	}
	return 0
}

func TestForwardXORReferenceMath(t *testing.T) {
	b := Backend[float32]{}
	layer := makeXORWeights()

	cases := []struct {
		input []float32
		want  []float32
	}{
		{[]float32{0, 0}, []float32{0, 0}},  // sum=0, sum=−1 → relu = 0,0
		{[]float32{1, 0}, []float32{1, 0}},  // sum=1, sum=0  → relu = 1,0
		{[]float32{0, 1}, []float32{1, 0}},
		{[]float32{1, 1}, []float32{2, 1}},  // sum=2, sum=1
	}
	for _, tc := range cases {
		got, err := b.Forward(layer, tc.input)
		if err != nil {
			t.Fatalf("Forward(%v): %v", tc.input, err)
		}
		for i := range got {
			if math.Abs(float64(got[i]-tc.want[i])) > float64(ToleranceF32) {
				t.Errorf("Forward(%v)[%d] = %v, want %v",
					tc.input, i, got[i], tc.want[i])
			}
		}
	}
}

func TestForwardErrorPaths(t *testing.T) {
	b := Backend[float32]{}
	layer := makeXORWeights()

	if _, err := b.Forward(compute.LayerHandle[float32]{Size: 2, Weights: [][]float32{{1}}}, []float32{1}); err == nil {
		t.Error("expected size mismatch error")
	}

	bad := layer
	bad.Activation = nil
	if _, err := b.Forward(bad, []float32{1, 1}); err == nil {
		t.Error("expected nil-activation error")
	}

	bad = layer
	bad.Weights = [][]float32{{1, 1}, {1, 1, 1}} // ragged
	if _, err := b.Forward(bad, []float32{1, 1}); err == nil {
		t.Error("expected ragged-weights error")
	}
}

func TestBackwardDistributesGradient(t *testing.T) {
	b := Backend[float32]{}
	layer := compute.LayerHandle[float32]{
		Size: 2,
		Weights: [][]float32{
			{0.5, -0.5},
			{1.0, 2.0},
		},
		Activation: relu[float32],
		Derivative: reluDeriv[float32],
	}
	gradient := []float32{1, 1}
	got, err := b.Backward(layer, gradient)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	// Hand-computed: dInput[0] = 1*0.5 + 1*1.0 = 1.5
	//                dInput[1] = 1*(-0.5) + 1*2.0 = 1.5
	want := []float32{1.5, 1.5}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > float64(ToleranceF32) {
			t.Errorf("Backward[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestBackwardErrorPaths(t *testing.T) {
	b := Backend[float32]{}
	layer := makeXORWeights()
	if _, err := b.Backward(layer, []float32{1}); err == nil {
		t.Error("expected gradient size error")
	}
	if _, err := b.Backward(compute.LayerHandle[float32]{Size: 0}, nil); err == nil {
		t.Error("expected empty-weights error")
	}
	ragged := compute.LayerHandle[float32]{
		Size:    2,
		Weights: [][]float32{{1, 1}, {1}},
	}
	if _, err := b.Backward(ragged, []float32{1, 1}); err == nil {
		t.Error("expected ragged error")
	}
}

func TestUpdateWeightsSGDStep(t *testing.T) {
	b := Backend[float32]{}
	layer := compute.LayerHandle[float32]{
		Size: 2,
		Weights: [][]float32{
			{1.0, 2.0},
			{3.0, 4.0},
		},
		Bias: []float32{0.5, 0.5},
	}
	inputs := []float32{1, 1}
	deltas := []float32{0.1, 0.2}
	rate := float32(0.5)

	if err := b.UpdateWeights(layer, inputs, deltas, rate); err != nil {
		t.Fatalf("UpdateWeights: %v", err)
	}
	// Expected: weights[i][j] -= rate*deltas[i]*inputs[j]
	// weights[0] = {1 - 0.5*0.1*1, 2 - 0.5*0.1*1} = {0.95, 1.95}
	// weights[1] = {3 - 0.5*0.2*1, 4 - 0.5*0.2*1} = {2.9, 3.9}
	wantW := [][]float32{{0.95, 1.95}, {2.9, 3.9}}
	for i := range wantW {
		for j := range wantW[i] {
			if math.Abs(float64(layer.Weights[i][j]-wantW[i][j])) > float64(ToleranceF32) {
				t.Errorf("weights[%d][%d] = %v, want %v",
					i, j, layer.Weights[i][j], wantW[i][j])
			}
		}
	}
	wantBias := []float32{0.5 - 0.05, 0.5 - 0.1}
	for i := range wantBias {
		if math.Abs(float64(layer.Bias[i]-wantBias[i])) > float64(ToleranceF32) {
			t.Errorf("bias[%d] = %v, want %v", i, layer.Bias[i], wantBias[i])
		}
	}
}

func TestUpdateWeightsErrorPaths(t *testing.T) {
	b := Backend[float32]{}
	layer := compute.LayerHandle[float32]{
		Size:    2,
		Weights: [][]float32{{1, 1}, {1, 1}},
		Bias:    []float32{0, 0},
	}
	if err := b.UpdateWeights(layer, []float32{1, 1}, []float32{0.1}, 0.1); err == nil {
		t.Error("expected delta-size error")
	}
	if err := b.UpdateWeights(layer, []float32{1}, []float32{0.1, 0.2}, 0.1); err == nil {
		t.Error("expected input-size error")
	}
}

func TestAllocateFree(t *testing.T) {
	b := Backend[float32]{}
	buf, err := b.Allocate(8)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if len(buf.Data) != 8 {
		t.Errorf("Allocate len = %d, want 8", len(buf.Data))
	}
	if err := b.Free(buf); err != nil {
		t.Errorf("Free: %v", err)
	}
	if _, err := b.Allocate(-1); err == nil || !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
	// Zero-sized allocation is permitted (empty slice).
	if zero, err := b.Allocate(0); err != nil || len(zero.Data) != 0 {
		t.Errorf("Allocate(0) = (%v, %v)", zero, err)
	}
}

func TestRegistration(t *testing.T) {
	got, err := compute.Get[float32](Name)
	if err != nil {
		t.Fatalf("Get(cpu): %v", err)
	}
	if got.Name() != "cpu" {
		t.Errorf("Name = %q", got.Name())
	}
	got64, err := compute.Get[float64](Name)
	if err != nil {
		t.Fatalf("Get[float64](cpu): %v", err)
	}
	if got64.Name() != "cpu" {
		t.Errorf("f64 Name = %q", got64.Name())
	}
}

func TestRegistryUnknown(t *testing.T) {
	_, err := compute.Get[float32]("does-not-exist")
	if err == nil || !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
}

func TestRegistryNamesIncludesCPU(t *testing.T) {
	names := compute.Names[float32]()
	if !slices.Contains(names, Name) {
		t.Errorf("Names() = %v, missing %q", names, Name)
	}
}

func TestF64Forward(t *testing.T) {
	b := Backend[float64]{}
	layer := compute.LayerHandle[float64]{
		Size:    1,
		Weights: [][]float64{{2.0, 3.0}},
		Bias:    []float64{1.0},
		Activation: func(v float64) float64 {
			return v // linear
		},
	}
	got, err := b.Forward(layer, []float64{0.5, 0.25})
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := 2.0*0.5 + 3.0*0.25 + 1.0
	if math.Abs(got[0]-want) > ToleranceF64 {
		t.Errorf("got %v, want %v", got[0], want)
	}
}

func TestForwardWithoutBias(t *testing.T) {
	b := Backend[float32]{}
	layer := compute.LayerHandle[float32]{
		Size:       1,
		Weights:    [][]float32{{2}},
		Activation: relu[float32],
	}
	got, err := b.Forward(layer, []float32{3})
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if got[0] != 6 {
		t.Errorf("got %v, want 6", got[0])
	}
}
