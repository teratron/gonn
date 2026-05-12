package norm

import (
	"encoding/json"
	"math"
	"testing"
)

const testEpsilon = 1e-4 // tolerance for floating-point comparisons

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < testEpsilon
}

// ─── BatchNorm Tests ────────────────────────────────────────────────────────

func TestBatchNormShapePreservation(t *testing.T) {
	cases := []struct {
		size int
		x    []float32
	}{
		{2, []float32{1, 2}},
		{4, []float32{1, 2, 3, 4}},
		{8, []float32{1, 2, 3, 4, 5, 6, 7, 8}},
	}
	for _, tc := range cases {
		bn := NewBatchNorm[float32](tc.size)
		out := bn.Forward(tc.x)
		if len(out) != len(tc.x) {
			t.Errorf("BatchNorm(%d).Forward: len=%d, want %d", tc.size, len(out), len(tc.x))
		}
	}
}

func TestBatchNormIdentityAffine(t *testing.T) {
	// With gamma=1 and beta=0 (default init), output should equal the
	// normalized values exactly.
	bn := NewBatchNorm[float32](4)
	x := []float32{1, 2, 3, 4}
	out := bn.Forward(x)
	if len(out) != 4 {
		t.Fatalf("unexpected output length %d", len(out))
	}
	// Manually compute expected normalized values.
	mean := float32(2.5)
	variance := float32(1.25)
	sd := float32(math.Sqrt(float64(variance + 1e-5)))
	for i, v := range x {
		want := (v - mean) / sd
		if !approxEqual(float64(out[i]), float64(want)) {
			t.Errorf("out[%d]=%v, want %v", i, out[i], want)
		}
	}
}

func TestBatchNormEMAUpdate(t *testing.T) {
	bn := NewBatchNorm[float32](4)
	// runningMean starts at 0, runningVar starts at 1 per NewBatchNorm.
	if bn.runningMean != 0 {
		t.Fatalf("initial runningMean want 0, got %v", bn.runningMean)
	}
	if bn.runningVar != 1 {
		t.Fatalf("initial runningVar want 1, got %v", bn.runningVar)
	}
	x := []float32{1, 2, 3, 4}
	bn.Forward(x)
	// batchMean=2.5, batchVar=1.25; momentum=0.1
	// newRunningMean = (1-0.1)*0 + 0.1*2.5 = 0.25
	wantMean := float32(0.25)
	// newRunningVar = (1-0.1)*1 + 0.1*1.25 = 1.025
	wantVar := float32(1.025)
	if !approxEqual(float64(bn.runningMean), float64(wantMean)) {
		t.Errorf("runningMean after forward: got %v, want %v", bn.runningMean, wantMean)
	}
	if !approxEqual(float64(bn.runningVar), float64(wantVar)) {
		t.Errorf("runningVar after forward: got %v, want %v", bn.runningVar, wantVar)
	}
}

func TestBatchNormEvalFrozenStats(t *testing.T) {
	bn := NewBatchNorm[float32](4)
	x := []float32{1, 2, 3, 4}
	// First forward in NormTrain — updates running stats.
	bn.Forward(x)
	gotMean := bn.runningMean
	gotVar := bn.runningVar

	// Switch to eval mode; stats must not change.
	bn.SetMode(NormEval)
	bn.Forward([]float32{10, 20, 30, 40})
	if bn.runningMean != gotMean {
		t.Errorf("runningMean changed in NormEval: before %v, after %v", gotMean, bn.runningMean)
	}
	if bn.runningVar != gotVar {
		t.Errorf("runningVar changed in NormEval: before %v, after %v", gotVar, bn.runningVar)
	}

	// Verify second eval-mode forward with same input as first gives identical output.
	bn2 := NewBatchNorm[float32](4)
	bn2.Forward(x)
	bn2.SetMode(NormEval)
	out1 := bn2.Forward(x)
	out2 := bn2.Forward(x)
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Errorf("eval output not deterministic at index %d: %v vs %v", i, out1[i], out2[i])
		}
	}
}

func TestBatchNormSingleSampleGuard(t *testing.T) {
	bn := NewBatchNorm[float32](1)
	x := []float32{42}
	out := bn.Forward(x)
	// Guard: single-element returns x unchanged.
	if len(out) != 1 || out[0] != x[0] {
		t.Errorf("single-sample guard: got %v, want %v", out, x)
	}
}

func TestBatchNormAffineDisabled(t *testing.T) {
	bn := NewBatchNorm[float32](4, WithBatchNormAffine[float32](false))
	g, b := bn.GradSlots()
	if g != nil || b != nil {
		t.Error("GradSlots with affine=false should return (nil, nil)")
	}
}

func TestBatchNormJSONRoundTrip(t *testing.T) {
	bn := NewBatchNorm[float32](4)
	bn.Forward([]float32{1, 2, 3, 4})
	bn.SetMode(NormEval)

	data, err := json.Marshal(bn)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}

	bn2 := &BatchNorm[float32]{}
	if err := json.Unmarshal(data, bn2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	if bn2.features != bn.features {
		t.Errorf("features: got %d, want %d", bn2.features, bn.features)
	}
	if bn2.runningMean != bn.runningMean {
		t.Errorf("runningMean: got %v, want %v", bn2.runningMean, bn.runningMean)
	}
	if bn2.runningVar != bn.runningVar {
		t.Errorf("runningVar: got %v, want %v", bn2.runningVar, bn.runningVar)
	}
	if NormMode(bn2.mode.Load()) != NormEval {
		t.Errorf("mode: got %v, want NormEval", bn2.mode.Load())
	}
	for i := range bn.gamma {
		if bn2.gamma[i] != bn.gamma[i] {
			t.Errorf("gamma[%d]: got %v, want %v", i, bn2.gamma[i], bn.gamma[i])
		}
	}
}

// ─── LayerNorm Tests ────────────────────────────────────────────────────────

func TestLayerNormShapePreservation(t *testing.T) {
	cases := [][]float32{
		{1, 2},
		{1, 2, 3, 4},
		{1, 2, 3, 4, 5, 6, 7, 8},
	}
	for _, x := range cases {
		ln := NewLayerNorm[float32](len(x))
		out := ln.Forward(x)
		if len(out) != len(x) {
			t.Errorf("LayerNorm(%d).Forward: len=%d, want %d", len(x), len(out), len(x))
		}
	}
}

func TestLayerNormAffineDisabled(t *testing.T) {
	ln := NewLayerNorm[float32](4, WithLayerNormAffine[float32](false))
	g, b := ln.GradSlots()
	if g != nil || b != nil {
		t.Error("GradSlots with affine=false should return (nil, nil)")
	}
	// With affine disabled, output should equal x_hat (no scale/shift).
	x := []float32{1, 2, 3, 4}
	out := ln.Forward(x)
	mean := float32(2.5)
	variance := float32(1.25)
	sd := float32(math.Sqrt(float64(variance + 1e-5)))
	for i, v := range x {
		want := (v - mean) / sd
		if !approxEqual(float64(out[i]), float64(want)) {
			t.Errorf("out[%d]=%v want %v", i, out[i], want)
		}
	}
}

func TestLayerNormModeNoEffect(t *testing.T) {
	ln := NewLayerNorm[float32](4)
	x := []float32{1, 2, 3, 4}
	out1 := ln.Forward(x)

	ln.SetMode(NormEval)
	out2 := ln.Forward(x)

	for i := range out1 {
		if out1[i] != out2[i] {
			t.Errorf("LayerNorm mode change affected output at index %d: %v vs %v", i, out1[i], out2[i])
		}
	}
}

func TestLayerNormJSONRoundTrip(t *testing.T) {
	ln := NewLayerNorm[float32](4)
	data, err := json.Marshal(ln)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	ln2 := &LayerNorm[float32]{}
	if err := json.Unmarshal(data, ln2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if ln2.features != ln.features {
		t.Errorf("features: got %d, want %d", ln2.features, ln.features)
	}
	if ln2.affine != ln.affine {
		t.Errorf("affine: got %v, want %v", ln2.affine, ln.affine)
	}
}

// ─── GroupNorm Tests ─────────────────────────────────────────────────────────

func TestGroupNormGroupPartition(t *testing.T) {
	// 4 features, 2 groups → groups of [x0,x1] and [x2,x3].
	gn, err := NewGroupNorm[float64](4, 2)
	if err != nil {
		t.Fatalf("NewGroupNorm: %v", err)
	}
	x := []float64{1, 2, 3, 4}
	out := gn.Forward(x)
	if len(out) != 4 {
		t.Fatalf("output length: got %d, want 4", len(out))
	}
	// Group 0: [1,2], mean=1.5, var=0.25, sd=sqrt(0.25+1e-5)
	sd0 := math.Sqrt(0.25 + 1e-5)
	wantOut0 := (1.0 - 1.5) / sd0
	wantOut1 := (2.0 - 1.5) / sd0
	if !approxEqual(out[0], wantOut0) {
		t.Errorf("out[0]=%v want %v", out[0], wantOut0)
	}
	if !approxEqual(out[1], wantOut1) {
		t.Errorf("out[1]=%v want %v", out[1], wantOut1)
	}
	// Group 1: [3,4], mean=3.5, var=0.25
	sd1 := math.Sqrt(0.25 + 1e-5)
	wantOut2 := (3.0 - 3.5) / sd1
	wantOut3 := (4.0 - 3.5) / sd1
	if !approxEqual(out[2], wantOut2) {
		t.Errorf("out[2]=%v want %v", out[2], wantOut2)
	}
	if !approxEqual(out[3], wantOut3) {
		t.Errorf("out[3]=%v want %v", out[3], wantOut3)
	}
}

func TestGroupNormGroupDivisibilityGuard(t *testing.T) {
	_, err := NewGroupNorm[float32](4, 3)
	if err == nil {
		t.Error("NewGroupNorm(4, 3): expected error, got nil")
	}
}

func TestGroupNormModeNoEffect(t *testing.T) {
	gn, _ := NewGroupNorm[float32](4, 2)
	x := []float32{1, 2, 3, 4}
	out1 := gn.Forward(x)
	gn.SetMode(NormEval)
	out2 := gn.Forward(x)
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Errorf("GroupNorm mode change affected output at index %d", i)
		}
	}
}

func TestGroupNormShapePreservation(t *testing.T) {
	gn, _ := NewGroupNorm[float32](8, 4)
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	out := gn.Forward(x)
	if len(out) != len(x) {
		t.Errorf("GroupNorm.Forward: len=%d, want %d", len(out), len(x))
	}
}

func TestGroupNormJSONRoundTrip(t *testing.T) {
	gn, _ := NewGroupNorm[float32](4, 2)
	data, err := json.Marshal(gn)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	gn2 := &GroupNorm[float32]{}
	if err := json.Unmarshal(data, gn2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if gn2.features != gn.features || gn2.groups != gn.groups {
		t.Errorf("features/groups mismatch after round-trip")
	}
}
