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

func TestBatchNormFirstForwardNearIdentity(t *testing.T) {
	// Virgin stats are mean=0, var=1 per feature, and train-mode Forward
	// normalizes with the PRE-update stats — so the very first forward is
	// x/sqrt(1+eps) (≈ identity with the default gamma=1, beta=0).
	bn := NewBatchNorm[float32](4)
	x := []float32{1, 2, 3, 4}
	out := bn.Forward(x)
	if len(out) != 4 {
		t.Fatalf("unexpected output length %d", len(out))
	}
	sd := float32(math.Sqrt(1 + 1e-5))
	for i, v := range x {
		want := v / sd
		if !approxEqual(float64(out[i]), float64(want)) {
			t.Errorf("out[%d]=%v, want %v", i, out[i], want)
		}
	}
}

func TestBatchNormEMAUpdate(t *testing.T) {
	bn := NewBatchNorm[float32](4)
	// Per-feature stats: runningMean starts at 0, runningVar at 1.
	for i := range 4 {
		if bn.runningMean[i] != 0 {
			t.Fatalf("initial runningMean[%d] want 0, got %v", i, bn.runningMean[i])
		}
		if bn.runningVar[i] != 1 {
			t.Fatalf("initial runningVar[%d] want 1, got %v", i, bn.runningVar[i])
		}
	}
	x := []float32{1, 2, 3, 4}
	bn.Forward(x)
	// momentum=0.1, sample-stream EMA per feature i:
	//   mean_i = 0.9·0 + 0.1·x_i
	//   var_i  = 0.9·1 + 0.1·(x_i − 0)²
	for i, v := range x {
		wantMean := 0.1 * v
		wantVar := 0.9 + 0.1*v*v
		if !approxEqual(float64(bn.runningMean[i]), float64(wantMean)) {
			t.Errorf("runningMean[%d]: got %v, want %v", i, bn.runningMean[i], wantMean)
		}
		if !approxEqual(float64(bn.runningVar[i]), float64(wantVar)) {
			t.Errorf("runningVar[%d]: got %v, want %v", i, bn.runningVar[i], wantVar)
		}
	}
}

func TestBatchNormEvalFrozenStats(t *testing.T) {
	bn := NewBatchNorm[float32](4)
	x := []float32{1, 2, 3, 4}
	// First forward in NormTrain — updates running stats.
	bn.Forward(x)
	gotMean := make([]float32, 4)
	gotVar := make([]float32, 4)
	copy(gotMean, bn.runningMean)
	copy(gotVar, bn.runningVar)

	// Switch to eval mode; stats must not change.
	bn.SetMode(NormEval)
	bn.Forward([]float32{10, 20, 30, 40})
	for i := range 4 {
		if bn.runningMean[i] != gotMean[i] {
			t.Errorf("runningMean[%d] changed in NormEval: before %v, after %v", i, gotMean[i], bn.runningMean[i])
		}
		if bn.runningVar[i] != gotVar[i] {
			t.Errorf("runningVar[%d] changed in NormEval: before %v, after %v", i, gotVar[i], bn.runningVar[i])
		}
	}

	// Verify second eval-mode forward with same input as first gives identical output.
	bn2 := NewBatchNorm[float32](4)
	bn2.Forward(x)
	bn2.SetMode(NormEval)
	out1 := bn2.Forward(x)
	got1 := make([]float32, len(out1))
	copy(got1, out1)
	out2 := bn2.Forward(x)
	for i := range got1 {
		if got1[i] != out2[i] {
			t.Errorf("eval output not deterministic at index %d: %v vs %v", i, got1[i], out2[i])
		}
	}
}

func TestBatchNormSingleFeatureWorks(t *testing.T) {
	// The pre-v0.11 whole-vector implementation degenerated on single-feature
	// layers; per-feature stats have no such failure mode.
	bn := NewBatchNorm[float32](1)
	x := []float32{42}
	out := bn.Forward(x)
	if len(out) != 1 {
		t.Fatalf("output length: got %d, want 1", len(out))
	}
	want := float32(42) / float32(math.Sqrt(1+1e-5))
	if !approxEqual(float64(out[0]), float64(want)) {
		t.Errorf("single-feature forward: got %v, want %v", out[0], want)
	}
}

// bnLoss computes L = Σ upstream_i · BatchNorm(x)_i on a FRESH layer so
// finite differences see a pure function of the perturbed argument.
func bnLoss(mk func() *BatchNorm[float64], x, upstream []float64) float64 {
	out := mk().Forward(x)
	var s float64
	for i, u := range upstream {
		s += u * out[i]
	}
	return s
}

// TestBatchNormBackwardFD verifies ∂L/∂x, ∂L/∂γ, ∂L/∂β via central
// differences. Exact because Forward normalizes with pre-update stats.
func TestBatchNormBackwardFD(t *testing.T) {
	const (
		n   = 5
		h   = 1e-5
		tol = 1e-4
	)
	x := []float64{0.5, -1.2, 0.3, 2.1, -0.8}
	upstream := []float64{1.0, -0.5, 0.3, 0.7, -0.2}
	mk := func() *BatchNorm[float64] {
		bn := NewBatchNorm[float64](n)
		for i := range bn.gamma {
			bn.gamma[i] = float64(i+1) * 0.3
		}
		return bn
	}

	bn := mk()
	bn.Forward(x)
	dx := bn.Backward(upstream)
	dg := make([]float64, n)
	db := make([]float64, n)
	copy(dg, bn.gammaGrad)
	copy(db, bn.betaGrad)

	for i := range x {
		xp := append([]float64(nil), x...)
		xm := append([]float64(nil), x...)
		xp[i] += h
		xm[i] -= h
		fd := (bnLoss(mk, xp, upstream) - bnLoss(mk, xm, upstream)) / (2 * h)
		if err := math.Abs(dx[i] - fd); err > tol {
			t.Errorf("∂L/∂x[%d]: analytic=%v FD=%v err=%v", i, dx[i], fd, err)
		}
	}
	for i := range x {
		mkP := func(delta float64, idx int) func() *BatchNorm[float64] {
			return func() *BatchNorm[float64] {
				bn := mk()
				bn.gamma[idx] += delta
				return bn
			}
		}
		fd := (bnLoss(mkP(h, i), x, upstream) - bnLoss(mkP(-h, i), x, upstream)) / (2 * h)
		if err := math.Abs(dg[i] - fd); err > tol {
			t.Errorf("∂L/∂γ[%d]: analytic=%v FD=%v err=%v", i, dg[i], fd, err)
		}
	}
	for i := range x {
		mkP := func(delta float64, idx int) func() *BatchNorm[float64] {
			return func() *BatchNorm[float64] {
				bn := mk()
				bn.beta[idx] += delta
				return bn
			}
		}
		fd := (bnLoss(mkP(h, i), x, upstream) - bnLoss(mkP(-h, i), x, upstream)) / (2 * h)
		if err := math.Abs(db[i] - fd); err > tol {
			t.Errorf("∂L/∂β[%d]: analytic=%v FD=%v err=%v", i, db[i], fd, err)
		}
	}
}

func TestBatchNormForwardInferencePure(t *testing.T) {
	bn := NewBatchNorm[float64](3)
	bn.Forward([]float64{1, 2, 3}) // move stats off the identity
	meanBefore := append([]float64(nil), bn.runningMean...)
	x := []float64{5, -2, 0.5}
	out1 := bn.ForwardInference(x)
	out2 := bn.ForwardInference(x)
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Errorf("ForwardInference not deterministic at %d", i)
		}
	}
	for i := range meanBefore {
		if bn.runningMean[i] != meanBefore[i] {
			t.Errorf("ForwardInference mutated runningMean[%d]", i)
		}
	}
}

func TestBatchNormAffineDisabled(t *testing.T) {
	bn := NewBatchNorm(4, WithBatchNormAffine[float32](false))
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
	for i := range bn.runningMean {
		if bn2.runningMean[i] != bn.runningMean[i] {
			t.Errorf("runningMean[%d]: got %v, want %v", i, bn2.runningMean[i], bn.runningMean[i])
		}
		if bn2.runningVar[i] != bn.runningVar[i] {
			t.Errorf("runningVar[%d]: got %v, want %v", i, bn2.runningVar[i], bn.runningVar[i])
		}
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
	ln := NewLayerNorm(4, WithLayerNormAffine[float32](false))
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

// ─── GroupNorm Backward Tests ────────────────────────────────────────────────

// TestGroupNormBackwardFD verifies ∂L/∂x through GroupNorm via central
// differences: 6 features in 2 groups, non-trivial gamma.
func TestGroupNormBackwardFD(t *testing.T) {
	const (
		n   = 6
		h   = 1e-5
		tol = 1e-4
	)
	x := []float64{0.5, -1.2, 0.3, 2.1, -0.8, 1.6}
	upstream := []float64{1.0, -0.5, 0.3, 0.7, -0.2, 0.9}
	mk := func() *GroupNorm[float64] {
		gn, err := NewGroupNorm[float64](n, 2)
		if err != nil {
			t.Fatalf("NewGroupNorm: %v", err)
		}
		for i := range gn.gamma {
			gn.gamma[i] = float64(i+1) * 0.3
		}
		return gn
	}
	gnLoss := func(xIn []float64) float64 {
		out := mk().Forward(xIn)
		var s float64
		for i, u := range upstream {
			s += u * out[i]
		}
		return s
	}

	gn := mk()
	gn.Forward(x)
	dx := gn.Backward(upstream)
	for i := range x {
		xp := append([]float64(nil), x...)
		xm := append([]float64(nil), x...)
		xp[i] += h
		xm[i] -= h
		fd := (gnLoss(xp) - gnLoss(xm)) / (2 * h)
		if err := math.Abs(dx[i] - fd); err > tol {
			t.Errorf("∂L/∂x[%d]: analytic=%v FD=%v err=%v", i, dx[i], fd, err)
		}
	}
}

// ─── LayerNorm Backward Tests ────────────────────────────────────────────────

// lnLoss computes the scalar L = Σ upstream_i * LayerNorm(x)_i used for FD checks.
func lnLoss(ln *LayerNorm[float64], x, upstream []float64) float64 {
	out := ln.Forward(x)
	var s float64
	for i, u := range upstream {
		s += u * float64(out[i])
	}
	return s
}

// TestLayerNorm_Backward verifies ∂L/∂x, ∂L/∂γ, ∂L/∂β via finite differences.
// Threshold: max_abs_err < 1e-4 for T=float64.
func TestLayerNorm_Backward(t *testing.T) {
	const (
		n   = 6
		h   = 1e-5 // FD step
		tol = 1e-4
	)

	x := []float64{0.5, -1.2, 0.3, 2.1, -0.8, 1.6}
	upstream := []float64{1.0, -0.5, 0.3, 0.7, -0.2, 0.9}

	// --- ∂L/∂x via FD ---
	ln := NewLayerNorm[float64](n)
	// Set non-trivial gamma so affine gradient is exercised.
	for i := range ln.gamma {
		ln.gamma[i] = float64(i+1) * 0.3
	}
	ln.Forward(x)
	dxAnalytic := ln.Backward(upstream)

	for i := range x {
		xp := make([]float64, n)
		copy(xp, x)
		xp[i] += h
		xm := make([]float64, n)
		copy(xm, x)
		xm[i] -= h

		var lp, lm float64
		// Use default gamma=1 in helper; we only compare shapes so use a fresh ln
		// with same gamma for FD.
		lnFD := NewLayerNorm[float64](n)
		for j := range lnFD.gamma {
			lnFD.gamma[j] = float64(j+1) * 0.3
		}
		lp = lnLoss(lnFD, xp, upstream)
		lnFD2 := NewLayerNorm[float64](n)
		for j := range lnFD2.gamma {
			lnFD2.gamma[j] = float64(j+1) * 0.3
		}
		lm = lnLoss(lnFD2, xm, upstream)

		dxFD := (lp - lm) / (2 * h)
		if err := math.Abs(dxAnalytic[i] - dxFD); err > tol {
			t.Errorf("∂L/∂x[%d]: analytic=%v FD=%v err=%v", i, dxAnalytic[i], dxFD, err)
		}
	}

	// --- ∂L/∂γ via FD ---
	lnG := NewLayerNorm[float64](n)
	for i := range lnG.gamma {
		lnG.gamma[i] = float64(i+1) * 0.3
	}
	lnG.Forward(x)
	lnG.Backward(upstream)
	dgAnalytic := make([]float64, n)
	copy(dgAnalytic, lnG.gammaGrad)

	for i := range x {
		lnp := NewLayerNorm[float64](n)
		lnm := NewLayerNorm[float64](n)
		for j := range lnp.gamma {
			lnp.gamma[j] = float64(j+1) * 0.3
			lnm.gamma[j] = float64(j+1) * 0.3
		}
		lnp.gamma[i] += h
		lnm.gamma[i] -= h
		lp := lnLoss(lnp, x, upstream)
		lm := lnLoss(lnm, x, upstream)
		dgFD := (lp - lm) / (2 * h)
		if err := math.Abs(dgAnalytic[i] - dgFD); err > tol {
			t.Errorf("∂L/∂γ[%d]: analytic=%v FD=%v err=%v", i, dgAnalytic[i], dgFD, err)
		}
	}

	// --- ∂L/∂β via FD ---
	lnB := NewLayerNorm[float64](n)
	lnB.Forward(x)
	lnB.Backward(upstream)
	dbAnalytic := make([]float64, n)
	copy(dbAnalytic, lnB.betaGrad)

	for i := range x {
		lnp := NewLayerNorm[float64](n)
		lnm := NewLayerNorm[float64](n)
		lnp.beta[i] += h
		lnm.beta[i] -= h
		lp := lnLoss(lnp, x, upstream)
		lm := lnLoss(lnm, x, upstream)
		dbFD := (lp - lm) / (2 * h)
		if err := math.Abs(dbAnalytic[i] - dbFD); err > tol {
			t.Errorf("∂L/∂β[%d]: analytic=%v FD=%v err=%v", i, dbAnalytic[i], dbFD, err)
		}
	}
}

// TestLayerNorm_BackwardAffineDisabled verifies Backward returns non-zero ∂L/∂x
// even when affine is disabled (no gamma/beta params to update).
func TestLayerNorm_BackwardAffineDisabled(t *testing.T) {
	ln := NewLayerNorm(4, WithLayerNormAffine[float64](false))
	x := []float64{1, 2, 3, 4}
	upstream := []float64{1, 1, 1, 1}
	ln.Forward(x)
	dx := ln.Backward(upstream)
	if len(dx) != 4 {
		t.Fatalf("Backward len: got %d, want 4", len(dx))
	}
	// Sum of gradients of a normalized output wrt constant upstream should be ~0.
	var sum float64
	for _, v := range dx {
		sum += v
	}
	if math.Abs(sum) > 1e-9 {
		t.Errorf("sum(∂L/∂x) with constant upstream should be ~0, got %v", sum)
	}
}

// TestLayerNorm_ForwardSeq verifies that ForwardSeq normalizes each position
// independently and matches standalone Forward calls.
func TestLayerNorm_ForwardSeq(t *testing.T) {
	const (
		features = 4
		seqLen   = 3
	)
	ln := NewLayerNorm[float64](features)
	ln2 := NewLayerNorm[float64](features)

	x := make([]float64, seqLen*features)
	for i := range x {
		x[i] = float64(i+1) * 0.1
	}

	got := ln.ForwardSeq(x, seqLen)
	if len(got) != seqLen*features {
		t.Fatalf("ForwardSeq len=%d want %d", len(got), seqLen*features)
	}

	for p := range seqLen {
		base := p * features
		want := ln2.Forward(x[base : base+features])
		for i := range features {
			if math.Abs(got[base+i]-want[i]) > 1e-12 {
				t.Errorf("pos %d feat %d: got %v want %v", p, i, got[base+i], want[i])
			}
		}
	}
}

// TestLayerNorm_BackwardSeqFD verifies BackwardSeq ∂L/∂x via finite differences.
func TestLayerNorm_BackwardSeqFD(t *testing.T) {
	const (
		features = 4
		seqLen   = 3
		h        = 1e-5
		tol      = 1e-4
	)

	x := make([]float64, seqLen*features)
	for i := range x {
		x[i] = float64(i+1) * 0.1
	}
	upstream := make([]float64, seqLen*features)
	for i := range upstream {
		upstream[i] = 1.0
	}

	ln := NewLayerNorm[float64](features)
	ln.ForwardSeq(x, seqLen)
	dxAnalytic := ln.BackwardSeq(upstream, seqLen)

	lnLossSeq := func(xIn []float64) float64 {
		l := NewLayerNorm[float64](features)
		out := l.ForwardSeq(xIn, seqLen)
		var s float64
		for i, u := range upstream {
			s += u * out[i]
		}
		return s
	}

	for i := range x {
		xp := make([]float64, len(x))
		xm := make([]float64, len(x))
		copy(xp, x)
		copy(xm, x)
		xp[i] += h
		xm[i] -= h
		fd := (lnLossSeq(xp) - lnLossSeq(xm)) / (2 * h)
		if err := math.Abs(dxAnalytic[i] - fd); err > tol {
			t.Errorf("BackwardSeq ∂L/∂x[%d]: analytic=%v FD=%v err=%v", i, dxAnalytic[i], fd, err)
		}
	}
}

// TestLayerNorm_ApplyGradSGD verifies that ApplyGradSGD updates gamma/beta and
// zeros the gradient buffers.
func TestLayerNorm_ApplyGradSGD(t *testing.T) {
	ln := NewLayerNorm[float64](3)
	x := []float64{1, 2, 3}
	upstream := []float64{1, 0, -1}
	ln.Forward(x)
	ln.Backward(upstream)

	gammaGradBefore := make([]float64, 3)
	copy(gammaGradBefore, ln.gammaGrad)
	gammaBefore := make([]float64, 3)
	copy(gammaBefore, ln.gamma)

	lr := 0.1
	ln.ApplyGradSGD(lr)

	for i := range ln.gamma {
		want := gammaBefore[i] - lr*gammaGradBefore[i]
		if math.Abs(ln.gamma[i]-want) > 1e-12 {
			t.Errorf("gamma[%d] after SGD: got %v want %v", i, ln.gamma[i], want)
		}
		if ln.gammaGrad[i] != 0 {
			t.Errorf("gammaGrad[%d] not zeroed after ApplyGradSGD", i)
		}
		if ln.betaGrad[i] != 0 {
			t.Errorf("betaGrad[%d] not zeroed after ApplyGradSGD", i)
		}
	}
}
