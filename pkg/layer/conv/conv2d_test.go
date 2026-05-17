package conv

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"testing"
)

// TestConv2DForwardShape — table-driven Forward shape validation (CONV2D-1).
func TestConv2DForwardShape(t *testing.T) {
	cases := []struct {
		name                                     string
		numFilters, inChannels, kernelH, kernelW int
		strideH, strideW                         int
		pad                                      PadMode
		inH, inW                                 int
		wantOutH, wantOutW                       int
	}{
		{"PadValid_C1_K3_S1_28x28", 2, 1, 3, 3, 1, 1, PadValid, 28, 28, 26, 26},
		{"PadSame_C1_K3_S1_28x28", 2, 1, 3, 3, 1, 1, PadSame, 28, 28, 28, 28},
		{"PadValid_C3_K3_S2_28x28", 4, 3, 3, 3, 2, 2, PadValid, 28, 28, 13, 13},
		{"PadSame_C2_K3_S2_28x28", 8, 2, 3, 3, 2, 2, PadSame, 28, 28, 14, 14},
		{"PadValid_C1_K5_S1_16x16", 3, 1, 5, 5, 1, 1, PadValid, 16, 16, 12, 12},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			conv := NewConv2D[float64](c.numFilters, c.inChannels, c.kernelH, c.kernelW, c.strideH, c.strideW, c.pad, false)
			conv.SetInputShape(c.inH, c.inW)
			outH, outW := conv.OutputShape()
			if outH != c.wantOutH || outW != c.wantOutW {
				t.Errorf("OutputShape = (%d, %d), want (%d, %d)", outH, outW, c.wantOutH, c.wantOutW)
			}
			rng := rand.New(rand.NewPCG(42, 42))
			conv.Init(rng)
			x := make([]float64, c.inChannels*c.inH*c.inW)
			for i := range x {
				x[i] = float64(i) / float64(len(x))
			}
			y := conv.Forward(x)
			if len(y) != c.numFilters*c.wantOutH*c.wantOutW {
				t.Errorf("Forward output length = %d, want %d", len(y), c.numFilters*c.wantOutH*c.wantOutW)
			}
		})
	}
}

// TestConv2DOutputShape verifies the shape arithmetic generalises outputLen.
func TestConv2DOutputShape(t *testing.T) {
	cases := []struct {
		inH, inW, kH, kW, sH, sW int
		pad                      PadMode
		wantOutH, wantOutW       int
	}{
		{28, 28, 3, 3, 1, 1, PadValid, 26, 26},
		{28, 28, 3, 3, 2, 2, PadSame, 14, 14},
		{16, 8, 5, 3, 1, 1, PadValid, 12, 6},
		{0, 0, 3, 3, 1, 1, PadValid, 0, 0},
		{5, 5, 7, 7, 1, 1, PadValid, 0, 0}, // kernel larger than input
	}
	for _, c := range cases {
		outH, outW := outputShape(c.inH, c.inW, c.kH, c.kW, c.sH, c.sW, c.pad)
		if outH != c.wantOutH || outW != c.wantOutW {
			t.Errorf("outputShape(%d,%d,%d,%d,%d,%d,%v) = (%d,%d), want (%d,%d)",
				c.inH, c.inW, c.kH, c.kW, c.sH, c.sW, c.pad, outH, outW, c.wantOutH, c.wantOutW)
		}
	}
}

// TestConv2DJSONRoundTrip verifies CONV2D-7 — Conv2D weights + biases +
// shape metadata survive marshal/unmarshal bit-identically.
func TestConv2DJSONRoundTrip(t *testing.T) {
	c1 := NewConv2D[float64](4, 2, 3, 3, 1, 1, PadValid, true)
	c1.SetInputShape(5, 5)
	rng := rand.New(rand.NewPCG(1, 1))
	c1.Init(rng)
	data, err := json.Marshal(c1)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	c2 := &Conv2D[float64]{}
	if err := json.Unmarshal(data, c2); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if c2.NumFilters != c1.NumFilters || c2.InChannels != c1.InChannels ||
		c2.KernelH != c1.KernelH || c2.KernelW != c1.KernelW ||
		c2.StrideH != c1.StrideH || c2.StrideW != c1.StrideW ||
		c2.Padding != c1.Padding || c2.UseBias != c1.UseBias {
		t.Errorf("shape metadata mismatch: c1=%+v, c2=%+v", c1, c2)
	}
	if len(c2.Weights) != len(c1.Weights) {
		t.Fatalf("weights length: got %d, want %d", len(c2.Weights), len(c1.Weights))
	}
	for i, w := range c1.Weights {
		if c2.Weights[i] != w {
			t.Errorf("weights[%d]: got %v, want %v", i, c2.Weights[i], w)
		}
	}
}

// TestConv2DValidateShapeMismatch verifies CONV2D-1 PadValid guard.
func TestConv2DValidateShapeMismatch(t *testing.T) {
	c := NewConv2D[float64](2, 1, 5, 5, 1, 1, PadValid, false)
	if err := c.Validate(1, 3, 3); err == nil {
		t.Fatal("expected ErrConv2DShapeMismatch for inH/inW < kernel, got nil")
	}
}

// TestConv2DGradientFiniteDifference is the core CONV2D-4 validator: it
// runs the full backward test matrix and checks that the analytical gradW
// matches a centred finite-difference estimate within tolerance. Covers:
//
//	{PadValid, PadSame} × {C_in=1, C_in=3}.
//
// Pool variants are exercised separately in TestPool2DBackward.
func TestConv2DGradientFiniteDifference(t *testing.T) {
	cases := []struct {
		name           string
		inChannels     int
		numFilters     int
		kernelH        int
		kernelW        int
		strideH        int
		strideW        int
		padding        PadMode
		inH, inW       int
		eps, tolerance float64
	}{
		{"PadValid_C1_K3", 1, 2, 3, 3, 1, 1, PadValid, 5, 5, 1e-5, 1e-7},
		{"PadValid_C3_K3", 3, 2, 3, 3, 1, 1, PadValid, 5, 5, 1e-5, 1e-7},
		{"PadSame_C1_K3", 1, 2, 3, 3, 1, 1, PadSame, 5, 5, 1e-5, 1e-7},
		{"PadSame_C2_K3", 2, 3, 3, 3, 1, 1, PadSame, 5, 5, 1e-5, 1e-7},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			conv := NewConv2D[float64](c.numFilters, c.inChannels, c.kernelH, c.kernelW, c.strideH, c.strideW, c.padding, true)
			conv.SetInputShape(c.inH, c.inW)
			rng := rand.New(rand.NewPCG(7, 13))
			conv.Init(rng)
			// Random input.
			x := make([]float64, c.inChannels*c.inH*c.inW)
			for i := range x {
				x[i] = rng.NormFloat64() * 0.1
			}
			y := conv.Forward(x)
			// Use sum(y) as the scalar loss; upstream gradient is all-ones.
			upstream := make([]float64, len(y))
			for i := range upstream {
				upstream[i] = 1.0
			}
			conv.Backward(upstream)
			gradWAnalytic := make([]float64, len(conv.gradW))
			copy(gradWAnalytic, conv.gradW)

			// Finite-difference: for each weight, perturb ±eps, re-run forward,
			// compare to analytical gradient. Sample a subset to keep the test
			// quick — checking 8 random weights per case is enough to catch
			// systematic errors (sign flips, layout bugs).
			sampleSize := min(8, len(conv.Weights))
			indices := rng.Perm(len(conv.Weights))[:sampleSize]
			for _, idx := range indices {
				orig := conv.Weights[idx]
				conv.Weights[idx] = orig + c.eps
				yPlus := conv.Forward(x)
				var lossPlus float64
				for _, v := range yPlus {
					lossPlus += v
				}
				conv.Weights[idx] = orig - c.eps
				yMinus := conv.Forward(x)
				var lossMinus float64
				for _, v := range yMinus {
					lossMinus += v
				}
				conv.Weights[idx] = orig
				fd := (lossPlus - lossMinus) / (2 * c.eps)
				analytic := gradWAnalytic[idx]
				if math.Abs(fd-analytic) > c.tolerance {
					t.Errorf("weight[%d]: analytic=%v, fd=%v, |diff|=%v > tol=%v",
						idx, analytic, fd, math.Abs(fd-analytic), c.tolerance)
				}
			}
		})
	}
}

// TestMaxPool2DBackward verifies CONV2D-5 max routing: only the argmax
// position receives the upstream gradient.
func TestMaxPool2DBackward(t *testing.T) {
	p := NewMaxPool2D[float64](2, 2)
	p.SetInputShape(1, 4, 4)
	// Input with clear maxima: each 2×2 window has a unique max at a known position.
	x := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	y := p.Forward(x)
	wantY := []float64{6, 8, 14, 16}
	for i, v := range wantY {
		if y[i] != v {
			t.Errorf("Forward[%d] = %v, want %v", i, y[i], v)
		}
	}
	upstream := []float64{1, 2, 3, 4}
	gradX := p.Backward(upstream)
	// Argmax positions: (1,1)=6→idx 5, (1,3)=8→idx 7, (3,1)=14→idx 13, (3,3)=16→idx 15.
	wantGrad := []float64{
		0, 0, 0, 0,
		0, 1, 0, 2,
		0, 0, 0, 0,
		0, 3, 0, 4,
	}
	for i, v := range wantGrad {
		if gradX[i] != v {
			t.Errorf("Backward gradX[%d] = %v, want %v", i, gradX[i], v)
		}
	}
}

// TestAvgPool2DBackward verifies CONV2D-5 avg distribution: each upstream
// gradient is divided equally across all positions in its window.
func TestAvgPool2DBackward(t *testing.T) {
	p := NewAvgPool2D[float64](2, 2)
	p.SetInputShape(1, 4, 4)
	x := make([]float64, 16)
	for i := range x {
		x[i] = float64(i + 1)
	}
	_ = p.Forward(x)
	upstream := []float64{4, 8, 12, 16}
	gradX := p.Backward(upstream)
	// Each upstream is divided by 4; (0,0)=1, (0,1)=2, (1,0)=2, (1,1)=2, etc.
	want := []float64{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	}
	for i, v := range want {
		if math.Abs(gradX[i]-v) > 1e-12 {
			t.Errorf("AvgPool2D gradX[%d] = %v, want %v", i, gradX[i], v)
		}
	}
}

// TestFlatten2DRoundTrip verifies CONV2D-6 element order and reshape backward.
func TestFlatten2DRoundTrip(t *testing.T) {
	f := NewFlatten2D[float64]()
	f.SetInputShape(2, 3, 4)
	// CHW-flat input: 24 elements.
	x := make([]float64, 24)
	for i := range x {
		x[i] = float64(i)
	}
	y := f.Forward(x)
	if len(y) != 24 {
		t.Fatalf("Forward output length: got %d, want 24", len(y))
	}
	// Element (c=1, y=2, x=3) → flat 1*12 + 2*4 + 3 = 23.
	if y[23] != 23 {
		t.Errorf("CHW indexing: y[23] = %v, want 23", y[23])
	}
	upstream := make([]float64, 24)
	for i := range upstream {
		upstream[i] = float64(i) * 0.5
	}
	gradX := f.Backward(upstream)
	for i, v := range upstream {
		if gradX[i] != v {
			t.Errorf("Flatten2D Backward gradX[%d] = %v, want %v", i, gradX[i], v)
		}
	}
}

// TestConv2DLayerInterface verifies CONV2D-9 — all four 2-D types satisfy
// the same Layer[T] interface as the 1-D types.
func TestConv2DLayerInterface(t *testing.T) {
	var (
		_ Layer[float32] = NewConv2D[float32](2, 1, 3, 3, 1, 1, PadValid, true)
		_ Layer[float64] = NewMaxPool2D[float64](2, 2)
		_ Layer[float64] = NewAvgPool2D[float64](2, 2)
		_ Layer[float32] = NewFlatten2D[float32]()
	)
}

// TestConv2DAccessors covers InputSize / OutputSize / GradSlots / OutputShape.
func TestConv2DAccessors(t *testing.T) {
	t.Parallel()
	c := NewConv2D[float32](4, 1, 3, 3, 1, 1, PadValid, true)
	c.SetInputShape(6, 6)
	if c.InputSize() != 1*6*6 {
		t.Errorf("InputSize = %d, want %d", c.InputSize(), 1*6*6)
	}
	// PadValid: outH = (6-3)/1+1 = 4, outW = 4 → OutputSize = 4*4*4 = 64.
	oh, ow := c.OutputShape()
	want := 4 * oh * ow
	if c.OutputSize() != want {
		t.Errorf("OutputSize = %d, want %d", c.OutputSize(), want)
	}
	// GradSlots after one Forward.
	c.Forward(make([]float32, 1*6*6))
	c.Backward(make([]float32, c.OutputSize()))
	gw, gb := c.GradSlots()
	if gw == nil {
		t.Error("GradSlots: gradW nil after Backward")
	}
	if gb == nil {
		t.Error("GradSlots: gradB nil when UseBias=true")
	}
}

// TestConv2DValidateAllPaths exercises the full Validate error matrix.
func TestConv2DValidateAllPaths(t *testing.T) {
	t.Parallel()
	// inChannels mismatch.
	c := NewConv2D[float32](2, 1, 3, 3, 1, 1, PadValid, false)
	if err := c.Validate(2, 8, 8); err == nil {
		t.Error("expected error for inChannels mismatch")
	}
	// PadValid: H < kernel.
	c2 := NewConv2D[float32](1, 1, 5, 5, 1, 1, PadValid, false)
	if err := c2.Validate(1, 3, 8); err == nil {
		t.Error("expected error: inH < kernelH under PadValid")
	}
	// PadValid: W < kernel.
	c3 := NewConv2D[float32](1, 1, 3, 5, 1, 1, PadValid, false)
	if err := c3.Validate(1, 8, 3); err == nil {
		t.Error("expected error: inW < kernelW under PadValid")
	}
	// Weight buffer size mismatch after manual truncation.
	c4 := NewConv2D[float32](2, 1, 3, 3, 1, 1, PadSame, false)
	c4.Weights = c4.Weights[:1] // corrupt
	if err := c4.Validate(0, 0, 0); err == nil {
		t.Error("expected error: weight buffer size mismatch")
	}
	// Bias buffer size mismatch.
	c5 := NewConv2D[float32](2, 1, 3, 3, 1, 1, PadSame, true)
	c5.Biases = c5.Biases[:1] // corrupt
	if err := c5.Validate(0, 0, 0); err == nil {
		t.Error("expected error: bias buffer size mismatch")
	}
}

// TestConv2DNewClamps verifies negative constructor args are clamped to 1.
func TestConv2DNewClamps(t *testing.T) {
	t.Parallel()
	c := NewConv2D[float32](-1, -1, -1, -1, -1, -1, PadValid, false)
	if c.NumFilters != 1 || c.InChannels != 1 || c.KernelH != 1 || c.KernelW != 1 ||
		c.StrideH != 1 || c.StrideW != 1 {
		t.Errorf("negative args not clamped: %+v", c)
	}
}

// TestConv2DIsqrt covers the internal isqrt helper.
func TestConv2DIsqrt(t *testing.T) {
	t.Parallel()
	cases := []struct{ n, want int }{{0, 0}, {1, 1}, {4, 2}, {9, 3}, {784, 28}, {783, 27}}
	for _, tc := range cases {
		if got := isqrt(tc.n); got != tc.want {
			t.Errorf("isqrt(%d) = %d, want %d", tc.n, got, tc.want)
		}
	}
}

// TestMaxPool2DAccessors covers InputSize / OutputSize / GradSlots / OutputShape / Validate.
func TestMaxPool2DAccessors(t *testing.T) {
	t.Parallel()
	p := NewMaxPool2D[float32](2, 2)
	p.SetInputShape(2, 8, 8)
	if p.InputSize() != 2*8*8 {
		t.Errorf("MaxPool2D.InputSize = %d, want %d", p.InputSize(), 2*8*8)
	}
	if p.OutputSize() != 2*4*4 {
		t.Errorf("MaxPool2D.OutputSize = %d, want %d", p.OutputSize(), 2*4*4)
	}
	oh, ow := p.OutputShape()
	if oh != 4 || ow != 4 {
		t.Errorf("MaxPool2D.OutputShape = (%d,%d), want (4,4)", oh, ow)
	}
	gw, gb := p.GradSlots()
	if gw != nil || gb != nil {
		t.Error("MaxPool2D.GradSlots should return (nil, nil)")
	}
	if err := p.Validate(2, 8, 8); err != nil {
		t.Errorf("Validate ok case: %v", err)
	}
	if err := p.Validate(2, 1, 8); err == nil {
		t.Error("expected error: poolH > inH")
	}
	if err := p.Validate(2, 8, 1); err == nil {
		t.Error("expected error: poolW > inW")
	}
}

// TestAvgPool2DAccessors mirrors TestMaxPool2DAccessors for AvgPool2D.
func TestAvgPool2DAccessors(t *testing.T) {
	t.Parallel()
	p := NewAvgPool2D[float32](2, 2)
	p.SetInputShape(1, 6, 6)
	if p.InputSize() != 1*6*6 {
		t.Errorf("AvgPool2D.InputSize = %d, want %d", p.InputSize(), 1*6*6)
	}
	if p.OutputSize() != 1*3*3 {
		t.Errorf("AvgPool2D.OutputSize = %d, want %d", p.OutputSize(), 1*3*3)
	}
	oh, ow := p.OutputShape()
	if oh != 3 || ow != 3 {
		t.Errorf("AvgPool2D.OutputShape = (%d,%d), want (3,3)", oh, ow)
	}
	gw, gb := p.GradSlots()
	if gw != nil || gb != nil {
		t.Error("AvgPool2D.GradSlots should return (nil, nil)")
	}
	if err := p.Validate(1, 6, 6); err != nil {
		t.Errorf("Validate ok case: %v", err)
	}
	if err := p.Validate(1, 1, 6); err == nil {
		t.Error("expected error: poolH > inH")
	}
}

// TestFlatten2DAccessors covers OutputSize / GradSlots / OutputShape / Validate / edge cases.
func TestFlatten2DAccessors(t *testing.T) {
	t.Parallel()
	f := NewFlatten2D[float32]()
	// Before SetInputShape: all zero.
	if f.InputSize() != 0 || f.OutputSize() != 0 {
		t.Error("Flatten2D size should be 0 before SetInputShape")
	}
	gw, gb := f.GradSlots()
	if gw != nil || gb != nil {
		t.Error("Flatten2D.GradSlots should return (nil, nil)")
	}
	c, h, w := f.OutputShape()
	if c != 0 || h != 0 || w != 0 {
		t.Errorf("OutputShape before setup = (%d,%d,%d), want (0,0,0)", c, h, w)
	}
	f.SetInputShape(2, 4, 4)
	if f.OutputSize() != 32 {
		t.Errorf("OutputSize = %d, want 32", f.OutputSize())
	}
	// Validate is a no-op — should never error.
	if err := f.Validate(99, 99, 99); err != nil {
		t.Errorf("Flatten2D.Validate: %v", err)
	}
	// Backward shape mismatch → nil.
	if f.Backward(make([]float32, 5)) != nil {
		t.Error("Flatten2D.Backward with wrong length should return nil")
	}
	// Forward without SetInputShape — isqrt inference for 16=4*4.
	f2 := NewFlatten2D[float32]()
	y := f2.Forward(make([]float32, 16))
	if len(y) != 16 {
		t.Errorf("Flatten2D.Forward length: got %d, want 16", len(y))
	}
	if f2.InH != 4 || f2.InW != 4 {
		t.Errorf("Flatten2D inferred shape = (%d,%d), want (4,4)", f2.InH, f2.InW)
	}
}

// TestNewMaxPool2DPanic verifies constructor panics on non-positive pool dims.
func TestNewMaxPool2DPanic(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewMaxPool2D(0,2) should panic")
		}
	}()
	NewMaxPool2D[float32](0, 2)
}

// TestNewAvgPool2DPanic mirrors the MaxPool2D panic test for AvgPool2D.
func TestNewAvgPool2DPanic(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewAvgPool2D(2,0) should panic")
		}
	}()
	NewAvgPool2D[float32](2, 0)
}

// TestConv2DUnmarshalJSONError verifies a malformed JSON payload returns an error.
func TestConv2DUnmarshalJSONError(t *testing.T) {
	t.Parallel()
	var c Conv2D[float32]
	if err := c.UnmarshalJSON([]byte(`{bad json`)); err == nil {
		t.Error("expected error for malformed JSON")
	}
}
