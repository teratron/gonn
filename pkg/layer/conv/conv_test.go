package conv

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

// TestOutputLenValid covers CONV-1: PadValid formula floor((n-k)/s)+1.
func TestOutputLenValid(t *testing.T) {
	t.Parallel()
	cases := []struct {
		inLen, k, s int
		pad         PadMode
		want        int
	}{
		{10, 3, 1, PadValid, 8},
		{10, 3, 2, PadValid, 4},
		{10, 5, 1, PadValid, 6},
		{3, 3, 1, PadValid, 1},
		{2, 3, 1, PadValid, 0}, // inLen < k → 0
		{0, 3, 1, PadValid, 0}, // empty input
		{10, 3, 1, PadSame, 10},
		{10, 3, 2, PadSame, 5},
		{11, 3, 2, PadSame, 6},
	}
	for _, tc := range cases {
		got := outputLen(tc.inLen, tc.k, tc.s, tc.pad)
		if got != tc.want {
			t.Errorf("outputLen(%d,%d,%d,%v) = %d, want %d",
				tc.inLen, tc.k, tc.s, tc.pad, got, tc.want)
		}
	}
}

// TestConv1DZeroWeights verifies that a Conv1D with zero weights produces a
// zero output (sanity check on the cross-correlation loop).
func TestConv1DZeroWeights(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](2, 3, 1, PadValid, false)
	in := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	out := c.Forward(in)
	if len(out) != 2*6 {
		t.Fatalf("output length = %d, want 12", len(out))
	}
	for i, v := range out {
		if v != 0 {
			t.Errorf("out[%d] = %v, want 0 with zero weights", i, v)
		}
	}
}

// TestConv1DKnownKernel checks the first filter against a hand-computed
// cross-correlation for a known input + kernel.
func TestConv1DKnownKernel(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](1, 3, 1, PadValid, false)
	// Kernel: [1, 0, -1] — first-order central difference.
	c.Weights[0] = 1
	c.Weights[1] = 0
	c.Weights[2] = -1
	in := []float32{2, 4, 6, 8, 10}
	out := c.Forward(in)
	// Output: [2-6, 4-8, 6-10] = [-4, -4, -4].
	want := []float32{-4, -4, -4}
	if len(out) != len(want) {
		t.Fatalf("output length = %d, want %d", len(out), len(want))
	}
	for i, v := range out {
		if v != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, v, want[i])
		}
	}
}

// TestConv1DBackwardShape ensures Backward returns ∂L/∂X of length InLen.
func TestConv1DBackwardShape(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](2, 3, 1, PadValid, true)
	in := []float32{1, 1, 1, 1, 1, 1}
	out := c.Forward(in)
	gradX := c.Backward(make([]float32, len(out)))
	if len(gradX) != len(in) {
		t.Errorf("gradX length = %d, want %d", len(gradX), len(in))
	}
	gW, gB := c.GradSlots()
	if len(gW) != len(c.Weights) {
		t.Errorf("gradW length = %d, want %d", len(gW), len(c.Weights))
	}
	if len(gB) != c.NumFilters {
		t.Errorf("gradB length = %d, want %d", len(gB), c.NumFilters)
	}
}

// TestConv1DGradFiniteDifference verifies ∂L/∂W via central differences
// (CONV-4). The loss is sum(output^2) / 2 so ∂L/∂y = y.
func TestConv1DGradFiniteDifference(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(1, 2))
	c := NewConv1D[float64](2, 3, 1, PadValid, false)
	for i := range c.Weights {
		c.Weights[i] = rng.NormFloat64() * 0.5
	}
	in := []float64{0.1, 0.4, -0.2, 0.7, 0.0, 0.3, -0.5, 0.6}
	out := c.Forward(in)
	// Upstream = output (∂L/∂y for L = sum(y^2)/2).
	upstream := append([]float64(nil), out...)
	_ = c.Backward(upstream)
	analytical, _ := c.GradSlots()

	const eps = 1e-5
	for i := range c.Weights {
		orig := c.Weights[i]
		c.Weights[i] = orig + eps
		yp := c.Forward(in)
		c.Weights[i] = orig - eps
		ym := c.Forward(in)
		c.Weights[i] = orig
		var lp, lm float64
		for j := range yp {
			lp += 0.5 * yp[j] * yp[j]
			lm += 0.5 * ym[j] * ym[j]
		}
		num := (lp - lm) / (2 * eps)
		diff := math.Abs(num - analytical[i])
		if diff > 1e-4 {
			t.Errorf("∂L/∂W[%d] analytic=%v finite-diff=%v |Δ|=%v", i, analytical[i], num, diff)
		}
	}
}

// TestConv1DInit confirms weights are non-zero after Init and biases are
// zero-initialised.
func TestConv1DInit(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(42, 99))
	c := NewConv1D[float32](3, 5, 1, PadValid, true)
	c.Init(rng)
	allZero := true
	for _, w := range c.Weights {
		if w != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("weights all zero after Init — He-normal sampling failed")
	}
	for i, b := range c.Biases {
		if b != 0 {
			t.Errorf("bias[%d] = %v, want 0", i, b)
		}
	}
}

// TestConv1DValidate enforces shape constraints.
func TestConv1DValidate(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](2, 5, 1, PadValid, false)
	if err := c.Validate(10); err != nil {
		t.Errorf("Validate(10) = %v, want nil", err)
	}
	err := c.Validate(3)
	if err == nil || !errors.Is(err, utils.ErrConvShapeMismatch) {
		t.Errorf("Validate(3) = %v, want ErrConvShapeMismatch", err)
	}
}

// TestConv1DPadSame checks the PadSame output length equals ceil(inLen/stride).
func TestConv1DPadSame(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](1, 3, 1, PadSame, false)
	c.Weights[0] = 1
	c.Weights[1] = 1
	c.Weights[2] = 1
	in := []float32{1, 1, 1, 1, 1}
	out := c.Forward(in)
	if len(out) != 5 {
		t.Errorf("PadSame output length = %d, want 5", len(out))
	}
}

// TestConv1DJSONRoundTrip verifies CONV-7 (weight persistence).
func TestConv1DJSONRoundTrip(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(7, 11))
	c := NewConv1D[float32](3, 4, 2, PadValid, true)
	c.Init(rng)
	// Force a forward to set InLen so the round-trip preserves shape.
	_ = c.Forward(make([]float32, 16))
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	c2 := &Conv1D[float32]{}
	if err := json.Unmarshal(data, c2); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if c2.NumFilters != c.NumFilters || c2.KernelSize != c.KernelSize ||
		c2.Stride != c.Stride || c2.Padding != c.Padding || c2.UseBias != c.UseBias {
		t.Errorf("round-trip mismatch: %+v vs %+v", c, c2)
	}
	for i := range c.Weights {
		if c.Weights[i] != c2.Weights[i] {
			t.Errorf("weights[%d] %v != %v", i, c.Weights[i], c2.Weights[i])
		}
	}
}

// TestMaxPool1DForward checks max selection within non-overlapping windows.
func TestMaxPool1DForward(t *testing.T) {
	t.Parallel()
	p := NewMaxPool1D[float32](2)
	in := []float32{1, 3, 2, 5, 0, 4}
	out := p.Forward(in)
	want := []float32{3, 5, 4}
	if len(out) != len(want) {
		t.Fatalf("output length = %d, want %d", len(out), len(want))
	}
	for i, v := range out {
		if v != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, v, want[i])
		}
	}
}

// TestMaxPool1DBackward verifies the gradient routes to argmax positions only.
func TestMaxPool1DBackward(t *testing.T) {
	t.Parallel()
	p := NewMaxPool1D[float32](2)
	in := []float32{1, 3, 2, 5, 0, 4}
	_ = p.Forward(in)
	gradX := p.Backward([]float32{10, 20, 30})
	want := []float32{0, 10, 0, 20, 0, 30}
	if len(gradX) != len(want) {
		t.Fatalf("gradX length = %d, want %d", len(gradX), len(want))
	}
	for i, v := range gradX {
		if v != want[i] {
			t.Errorf("gradX[%d] = %v, want %v", i, v, want[i])
		}
	}
}

// TestAvgPool1DForward verifies average computation.
func TestAvgPool1DForward(t *testing.T) {
	t.Parallel()
	p := NewAvgPool1D[float32](2)
	in := []float32{2, 4, 6, 8}
	out := p.Forward(in)
	want := []float32{3, 7}
	for i, v := range out {
		if v != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, v, want[i])
		}
	}
}

// TestAvgPool1DBackward verifies even gradient distribution.
func TestAvgPool1DBackward(t *testing.T) {
	t.Parallel()
	p := NewAvgPool1D[float32](2)
	_ = p.Forward([]float32{2, 4, 6, 8})
	gradX := p.Backward([]float32{10, 20})
	want := []float32{5, 5, 10, 10}
	for i, v := range gradX {
		if v != want[i] {
			t.Errorf("gradX[%d] = %v, want %v", i, v, want[i])
		}
	}
}

// TestPoolValidate covers the size-mismatch error path.
func TestPoolValidate(t *testing.T) {
	t.Parallel()
	p := NewMaxPool1D[float32](5)
	if err := p.Validate(10); err != nil {
		t.Errorf("Validate(10) = %v, want nil", err)
	}
	err := p.Validate(3)
	if err == nil || !errors.Is(err, utils.ErrConvPoolSizeMismatch) {
		t.Errorf("Validate(3) = %v, want ErrConvPoolSizeMismatch", err)
	}
}

// TestFlattenIdentity checks Forward / Backward are identity on conformant shapes.
func TestFlattenIdentity(t *testing.T) {
	t.Parallel()
	f := NewFlatten[float32]()
	in := []float32{1, 2, 3, 4, 5}
	out := f.Forward(in)
	if len(out) != len(in) {
		t.Fatalf("Forward len = %d, want %d", len(out), len(in))
	}
	for i, v := range out {
		if v != in[i] {
			t.Errorf("Forward[%d] = %v, want %v", i, v, in[i])
		}
	}
	gradX := f.Backward(in)
	for i, v := range gradX {
		if v != in[i] {
			t.Errorf("Backward[%d] = %v, want %v", i, v, in[i])
		}
	}
	// Shape mismatch on Backward returns nil.
	if got := f.Backward([]float32{1, 2}); got != nil {
		t.Errorf("Backward(short) = %v, want nil", got)
	}
}

// TestLayerAccessors exercises the InputSize / OutputSize / GradSlots
// accessors across all three layer types after a Forward pass.
func TestLayerAccessors(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](2, 3, 1, PadValid, false)
	_ = c.Forward(make([]float32, 8))
	if c.InputSize() != 8 {
		t.Errorf("Conv1D.InputSize = %d, want 8", c.InputSize())
	}
	if c.OutputSize() != 12 {
		t.Errorf("Conv1D.OutputSize = %d, want 12", c.OutputSize())
	}

	mp := NewMaxPool1D[float32](2)
	_ = mp.Forward(make([]float32, 6))
	if mp.InputSize() != 6 || mp.OutputSize() != 3 {
		t.Errorf("MaxPool1D sizes = (%d,%d), want (6,3)", mp.InputSize(), mp.OutputSize())
	}
	if gw, gb := mp.GradSlots(); gw != nil || gb != nil {
		t.Errorf("MaxPool1D.GradSlots = (%v,%v), want (nil,nil)", gw, gb)
	}

	ap := NewAvgPool1D[float32](2)
	_ = ap.Forward(make([]float32, 6))
	if ap.InputSize() != 6 || ap.OutputSize() != 3 {
		t.Errorf("AvgPool1D sizes = (%d,%d), want (6,3)", ap.InputSize(), ap.OutputSize())
	}
	if gw, gb := ap.GradSlots(); gw != nil || gb != nil {
		t.Errorf("AvgPool1D.GradSlots = (%v,%v), want (nil,nil)", gw, gb)
	}

	f := NewFlatten[float32]()
	_ = f.Forward(make([]float32, 5))
	if f.InputSize() != 5 || f.OutputSize() != 5 {
		t.Errorf("Flatten sizes = (%d,%d), want (5,5)", f.InputSize(), f.OutputSize())
	}
	if gw, gb := f.GradSlots(); gw != nil || gb != nil {
		t.Errorf("Flatten.GradSlots = (%v,%v), want (nil,nil)", gw, gb)
	}
	if err := f.Validate(0); err != nil {
		t.Errorf("Flatten.Validate = %v, want nil", err)
	}
}

// TestAvgPoolValidate covers the size-mismatch error path on AvgPool1D.
func TestAvgPoolValidate(t *testing.T) {
	t.Parallel()
	p := NewAvgPool1D[float32](5)
	if err := p.Validate(10); err != nil {
		t.Errorf("Validate(10) = %v, want nil", err)
	}
	err := p.Validate(3)
	if err == nil || !errors.Is(err, utils.ErrConvPoolSizeMismatch) {
		t.Errorf("Validate(3) = %v, want ErrConvPoolSizeMismatch", err)
	}
}

// TestPoolPanics verifies that non-positive pool sizes panic the constructor.
func TestPoolPanics(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewMaxPool1D(0) did not panic")
		}
	}()
	_ = NewMaxPool1D[float32](0)
}

// TestAvgPoolPanics mirrors TestPoolPanics for AvgPool1D.
func TestAvgPoolPanics(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewAvgPool1D(-1) did not panic")
		}
	}()
	_ = NewAvgPool1D[float32](-1)
}

// TestConv1DConstructorClamp verifies non-positive shape args are clamped.
func TestConv1DConstructorClamp(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](0, -1, 0, PadValid, false)
	if c.NumFilters != 1 || c.KernelSize != 1 || c.Stride != 1 {
		t.Errorf("constructor clamp failed: %+v", c)
	}
}

// TestConv1DBackwardShapeMismatch ensures bad upstream length returns nil.
func TestConv1DBackwardShapeMismatch(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](2, 3, 1, PadValid, false)
	_ = c.Forward(make([]float32, 8))
	if got := c.Backward(make([]float32, 5)); got != nil {
		t.Errorf("Backward(short) = %v, want nil", got)
	}
}

// TestMaxPoolBackwardShapeMismatch ensures bad upstream length returns nil.
func TestMaxPoolBackwardShapeMismatch(t *testing.T) {
	t.Parallel()
	p := NewMaxPool1D[float32](2)
	_ = p.Forward(make([]float32, 6))
	if got := p.Backward(make([]float32, 5)); got != nil {
		t.Errorf("Backward(short) = %v, want nil", got)
	}
}

// TestConv1DValidateAll covers all Validate error branches.
func TestConv1DValidateAll(t *testing.T) {
	t.Parallel()
	c := NewConv1D[float32](2, 3, 1, PadValid, true)
	// Corrupt the weight slice — should trigger integrity error.
	c.Weights = c.Weights[:0]
	if err := c.Validate(10); err == nil {
		t.Error("Validate with empty weights returned nil")
	}
	c = NewConv1D[float32](2, 3, 1, PadValid, true)
	c.Biases = c.Biases[:0]
	if err := c.Validate(10); err == nil {
		t.Error("Validate with empty biases returned nil")
	}
}

// TestConv1DUnmarshalErrors covers corrupted JSON paths.
func TestConv1DUnmarshalErrors(t *testing.T) {
	t.Parallel()
	c := &Conv1D[float32]{}
	if err := c.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Error("UnmarshalJSON(bad-json) returned nil")
	}
	if err := c.UnmarshalJSON([]byte(`{"num_filters":0}`)); err == nil {
		t.Error("UnmarshalJSON(zero-filters) returned nil")
	}
	if err := c.UnmarshalJSON([]byte(`{"num_filters":2,"kernel_size":3,"stride":1,"weights":[]}`)); err == nil {
		t.Error("UnmarshalJSON(short-weights) returned nil")
	}
}

// TestPadSamePadding verifies the (left, right) split is correct.
func TestPadSamePadding(t *testing.T) {
	t.Parallel()
	l, r := padSamePadding(5, 3, 1)
	if l+r != 2 {
		t.Errorf("total padding = %d, want 2 (k=3, s=1, in=5 → out=5)", l+r)
	}
	l, r = padSamePadding(5, 4, 1)
	if l+r != 3 {
		t.Errorf("total padding = %d, want 3 (k=4, s=1, in=5 → out=5)", l+r)
	}
}
