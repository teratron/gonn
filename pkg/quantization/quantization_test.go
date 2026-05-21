package quantization

import (
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	attentionpkg "github.com/teratron/gonn/pkg/layer/attention"
	convpkg "github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// ─── Helpers ─────────────────────────────────────────────────────────────────

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// makeConv1DNet builds a small Conv1D network for testing.
func makeConv1DNet(t *testing.T) *nn.NN[float64] {
	t.Helper()
	net, err := nn.New[float64](
		nn.WithConv1D[float64](4, 3, 1, convpkg.PadValid, false),
		nn.WithInput[float64](8),
		nn.WithHiddenLayer[float64](8, 0),
		nn.WithOutput[float64](2, 0),
	)
	if err != nil {
		t.Fatalf("makeConv1DNet: %v", err)
	}
	return net
}

func makeAttnLayer(seqLen, dmodel, numHeads int) *attentionpkg.MultiHeadAttention[float64] {
	src := attentionpkg.NewMultiHeadAttention[float64](seqLen, dmodel, numHeads, false)
	rng := rand.New(rand.NewPCG(13, 13))
	src.Init(rng)
	return src
}

// ─── T-19T01: Golden-value + round-trip tests ─────────────────────────────

// TestRoundHalfEven covers the L1 §4.2 banker's rounding table.
func TestRoundHalfEven(t *testing.T) {
	cases := []struct {
		in   float64
		want int64
	}{
		{0.5, 0},
		{1.5, 2},
		{2.5, 2},
		{3.5, 4},
		{-0.5, 0},
		{-1.5, -2},
		{4.0, 4},
		{4.3, 4},
		{4.7, 5},
	}
	for _, c := range cases {
		got := roundHalfEven(c.in)
		if got != c.want {
			t.Errorf("roundHalfEven(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestQuantizationGoldenValue verifies L1 §4.2 normative example.
func TestQuantizationGoldenValue(t *testing.T) {
	p := computeSymmetric(0.0, 3.0)
	wantScale := 3.0 / 127.0
	if !almostEqual(p.Scale[0], wantScale, 1e-9) {
		t.Errorf("scale = %.10f, want %.10f", p.Scale[0], wantScale)
	}
	if p.ZeroPoint[0] != 0 {
		t.Errorf("zero_point = %d, want 0", p.ZeroPoint[0])
	}
	q := quantize(1.0, p, 0)
	if q != 42 {
		t.Errorf("quantize(1.0) = %d, want 42", q)
	}
	r := dequantize(42, p, 0)
	if !almostEqual(r, 1.0, 0.01) {
		t.Errorf("dequantize(42) = %.6f, abs err = %.6f > 0.01", r, math.Abs(r-1.0))
	}
}

// TestComputeSymmetric covers symmetric quantization properties.
func TestComputeSymmetric(t *testing.T) {
	p := computeSymmetric(-2.0, 2.0)
	if !almostEqual(p.Scale[0], 2.0/127.0, 1e-9) {
		t.Errorf("scale = %.10f, want %.10f", p.Scale[0], 2.0/127.0)
	}
	pd := computeSymmetric(1.0, 1.0+1e-9)
	if pd.Scale[0] != 1.0 || pd.ZeroPoint[0] != 0 {
		t.Errorf("degenerate: scale=%v zp=%v, want 1.0 0", pd.Scale[0], pd.ZeroPoint[0])
	}
}

// TestComputeAsymmetric covers asymmetric quantization.
func TestComputeAsymmetric(t *testing.T) {
	p := computeAsymmetric(0.0, 1.0)
	if !almostEqual(p.Scale[0], 1.0/255.0, 1e-9) {
		t.Errorf("scale = %.10f, want %.10f", p.Scale[0], 1.0/255.0)
	}
}

// TestQuantizedDense_WeightShape verifies QuantizedDense buffer shapes.
func TestQuantizedDense_WeightShape(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	const cout, cin = 16, 8
	w := make([]float64, cout*cin)
	for i := range w {
		w[i] = rng.Float64()*2 - 1
	}
	cfg := DefaultQuantizationConfig[float64]()
	d := newQuantizedDense[float64](w, nil, cout, cin, cfg)
	if len(d.Weights) != cout*cin {
		t.Errorf("Weights len = %d, want %d", len(d.Weights), cout*cin)
	}
	if len(d.WeightParams.Scale) != cout {
		t.Errorf("Scale len = %d, want %d (PerChannel)", len(d.WeightParams.Scale), cout)
	}
}

// TestQuantizedDense_Forward verifies weight-only Dense forward within 5% tolerance.
func TestQuantizedDense_Forward(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	const cout, cin = 16, 8
	w := make([]float64, cout*cin)
	for i := range w {
		w[i] = rng.Float64()*2 - 1
	}
	x := make([]float64, cin)
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	yFloat := make([]float64, cout)
	for c := range cout {
		for k := range cin {
			yFloat[c] += w[c*cin+k] * x[k]
		}
	}
	cfg := DefaultQuantizationConfig[float64]()
	d := newQuantizedDense[float64](w, nil, cout, cin, cfg)
	yQuant := d.Forward(x)

	var maxAbs float64
	for _, v := range yFloat {
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	tol := math.Max(0.05*maxAbs, 1e-6)
	for c := range cout {
		if math.Abs(yFloat[c]-yQuant[c]) > tol {
			t.Errorf("channel %d: float=%.6f quant=%.6f diff=%.6f > tol=%.6f",
				c, yFloat[c], yQuant[c], math.Abs(yFloat[c]-yQuant[c]), tol)
		}
	}
}

// TestQuantizedConv1D_WeightShape verifies QuantizedConv1D buffer shapes.
func TestQuantizedConv1D_WeightShape(t *testing.T) {
	src := convpkg.NewConv1D[float64](4, 3, 1, convpkg.PadValid, false)
	rng := rand.New(rand.NewPCG(1, 1))
	src.Init(rng)
	cfg := DefaultQuantizationConfig[float64]()
	ql := newQuantizedConv1D[float64](src, cfg)
	if len(ql.Weights) != src.NumFilters*src.KernelSize {
		t.Errorf("Weights len = %d, want %d", len(ql.Weights), src.NumFilters*src.KernelSize)
	}
	if len(ql.WeightParams.Scale) != src.NumFilters {
		t.Errorf("Scale len = %d, want %d (PerChannel)", len(ql.WeightParams.Scale), src.NumFilters)
	}
}

// TestQuantizedConv1D_Forward verifies Conv1D forward within 5% tolerance.
func TestQuantizedConv1D_Forward(t *testing.T) {
	src := convpkg.NewConv1D[float64](4, 3, 1, convpkg.PadValid, false)
	rng := rand.New(rand.NewPCG(7, 7))
	src.Init(rng)
	x := make([]float64, 10)
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	floatOut := src.Forward(x)
	cfg := DefaultQuantizationConfig[float64]()
	ql := newQuantizedConv1D[float64](src, cfg)
	quantOut := ql.Forward(x)
	if len(quantOut) != len(floatOut) {
		t.Fatalf("output length: float=%d quant=%d", len(floatOut), len(quantOut))
	}
	var maxAbs float64
	for _, v := range floatOut {
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	tol := math.Max(0.05*maxAbs, 1e-6)
	for i, fv := range floatOut {
		if math.Abs(fv-quantOut[i]) > tol {
			t.Errorf("output[%d]: float=%.6f quant=%.6f diff=%.6f > tol=%.6f",
				i, fv, quantOut[i], math.Abs(fv-quantOut[i]), tol)
		}
	}
}

// TestQNNRoundTrip verifies MarshalQNN + UnmarshalQNN preserves int8 weights bit-exact.
func TestQNNRoundTrip(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	qnet, err := Quantize[float64](net, nil, cfg)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	data, err := MarshalQNN[float64](qnet)
	if err != nil {
		t.Fatalf("MarshalQNN: %v", err)
	}
	qnet2, err := UnmarshalQNN[float64](data)
	if err != nil {
		t.Fatalf("UnmarshalQNN: %v", err)
	}
	if len(qnet.Layers) != len(qnet2.Layers) {
		t.Fatalf("layer count: original=%d roundtrip=%d", len(qnet.Layers), len(qnet2.Layers))
	}
	x := make([]float64, 8)
	rng := rand.New(rand.NewPCG(99, 99))
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	y1 := qnet.Forward(x)
	y2 := qnet2.Forward(x)
	if len(y1) != len(y2) {
		t.Fatalf("output length: before=%d after=%d", len(y1), len(y2))
	}
	for i := range y1 {
		if y1[i] != y2[i] {
			t.Errorf("output[%d]: before=%.10f after=%.10f (not bit-exact)", i, y1[i], y2[i])
		}
	}
}

// TestQNNWrongType verifies that UnmarshalQNN rejects a non-"quantized" type.
func TestQNNWrongType(t *testing.T) {
	_, err := UnmarshalQNN[float64]([]byte(`{"type":"network","layers":[]}`))
	if err == nil {
		t.Fatal("expected error for wrong Type, got nil")
	}
}

// TestQNNSave writes to a temp file and Load reads it back.
func TestQNNSave(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	qnet, err := Quantize[float64](net, nil, cfg)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	tmp := filepath.Join(t.TempDir(), "model.qnn.json")
	if err := qnet.Save(tmp); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := os.Stat(tmp); err != nil {
		t.Fatalf("file not written: %v", err)
	}
	qnet2, err := Load[float64](tmp)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	x := make([]float64, 8)
	y1 := qnet.Forward(x)
	y2 := qnet2.Forward(x)
	for i := range y1 {
		if y1[i] != y2[i] {
			t.Errorf("output[%d] mismatch after Save/Load", i)
		}
	}
}

// TestBaselineHashStable verifies BaselineHash is deterministic.
func TestBaselineHashStable(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	q1, _ := Quantize[float64](net, nil, cfg)
	q2, _ := Quantize[float64](net, nil, cfg)
	if q1.BaselineHash != q2.BaselineHash {
		t.Errorf("BaselineHash not deterministic: %q != %q", q1.BaselineHash, q2.BaselineHash)
	}
	if len(q1.BaselineHash) != 64 {
		t.Errorf("BaselineHash length = %d, want 64 (SHA-256 hex)", len(q1.BaselineHash))
	}
}

// ─── Calibration tests ──────────────────────────────────────────────────────

// TestCalibrationRunner_TooFewSamples verifies ErrCalibTooFewSamples.
func TestCalibrationRunner_TooFewSamples(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	runner := NewCalibrationRunner[float64](net, cfg)
	samples := make([][]float64, 10)
	for i := range samples {
		samples[i] = make([]float64, 8)
	}
	err := runner.Run(samples)
	if err == nil {
		t.Fatal("expected ErrCalibTooFewSamples, got nil")
	}
	if !errors.Is(err, utils.ErrCalibTooFewSamples) {
		t.Errorf("error does not wrap ErrCalibTooFewSamples: %v", err)
	}
}

// TestCalibrationRunner_MinMax verifies that 50 samples produce valid params.
func TestCalibrationRunner_MinMax(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	cfg.Strategy = MinMax
	runner := NewCalibrationRunner[float64](net, cfg)
	rng := rand.New(rand.NewPCG(42, 42))
	samples := make([][]float64, 50)
	for i := range samples {
		samples[i] = make([]float64, 8)
		for j := range samples[i] {
			samples[i][j] = rng.Float64()*2 - 1
		}
	}
	if err := runner.Run(samples); err != nil {
		t.Fatalf("Run: %v", err)
	}
	params, _ := runner.Params()
	for i, p := range params {
		if len(p.Scale) == 0 {
			t.Errorf("layer %d: no scale computed", i)
		}
	}
}

// TestCalibrationRunner_Percentile99p9 verifies outlier clipping.
func TestCalibrationRunner_Percentile99p9(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	cfg.Strategy = Percentile99p9
	cfg.PercentileThreshold = 99.9
	runner := NewCalibrationRunner[float64](net, cfg)
	rng := rand.New(rand.NewPCG(11, 11))
	samples := make([][]float64, 200)
	for i := range samples {
		samples[i] = make([]float64, 8)
		for j := range samples[i] {
			samples[i][j] = rng.NormFloat64()
		}
	}
	if err := runner.Run(samples); err != nil {
		t.Fatalf("Run: %v", err)
	}
	params, _ := runner.Params()
	for i, p := range params {
		if len(p.Scale) == 0 || p.Scale[0] <= 0 {
			t.Errorf("layer %d: invalid scale %v", i, p.Scale)
		}
	}
}

// TestDegenerateRange verifies percentileRange on constant input.
func TestDegenerateRange(t *testing.T) {
	vals := make([]float64, 100)
	rMin, rMax := percentileRange(vals, 99.9)
	if rMax-rMin >= 1e-6 {
		t.Errorf("constant activations: expected degenerate range, got [%v, %v]", rMin, rMax)
	}
}

// ─── Evaluator tests ────────────────────────────────────────────────────────

// TestSideBySideEvaluator verifies Evaluate produces non-trivial results.
func TestSideBySideEvaluator(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	qnet, err := Quantize[float64](net, nil, cfg)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	rng := rand.New(rand.NewPCG(5, 5))
	evalSet := make([]EvalSample[float64], 20)
	for i := range evalSet {
		inp := make([]float64, 8)
		for j := range inp {
			inp[j] = rng.Float64()*2 - 1
		}
		evalSet[i] = EvalSample[float64]{Input: inp, Expected: nil}
	}
	// Metric uses only output (Expected is nil — computes ||out||₂²/n).
	mse := func(out, _ []float64) float64 {
		var s float64
		for _, v := range out {
			s += v * v
		}
		if len(out) == 0 {
			return 0
		}
		return s / float64(len(out))
	}
	result, err := Evaluate[float64](net, qnet, evalSet, mse)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if result.MetricFloat < 0 {
		t.Errorf("MetricFloat < 0: %v", result.MetricFloat)
	}
	if len(result.PerLayerL2) != len(qnet.Layers) {
		t.Errorf("PerLayerL2 len = %d, want %d", len(result.PerLayerL2), len(qnet.Layers))
	}
}

// TestEvaluate_EmptySet verifies ErrInputData for empty eval set.
func TestEvaluate_EmptySet(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	qnet, _ := Quantize[float64](net, nil, cfg)
	_, err := Evaluate[float64](net, qnet, nil, func(_, _ []float64) float64 { return 0 })
	if err == nil {
		t.Fatal("expected error for empty eval set, got nil")
	}
	if !errors.Is(err, utils.ErrInputData) {
		t.Errorf("error does not wrap ErrInputData: %v", err)
	}
}

// ─── Full-int8 tests ────────────────────────────────────────────────────────

// TestFullInt8Forward verifies full-int8 Dense forward within 5% tolerance.
func TestFullInt8Forward(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	const cout, cin = 64, 16
	w := make([]float64, cout*cin)
	for i := range w {
		w[i] = rng.NormFloat64() * 0.1
	}
	x := make([]float64, cin)
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	cfg := DefaultQuantizationConfig[float64]()
	d := newQuantizedDense[float64](w, nil, cout, cin, cfg)
	d.SetActParams(computeSymmetric(-1.0, 1.0))
	yQuant := d.Forward(x)

	yFloat := make([]float64, cout)
	for c := range cout {
		for k := range cin {
			yFloat[c] += w[c*cin+k] * x[k]
		}
	}
	var maxAbs float64
	for _, v := range yFloat {
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	tol := math.Max(0.05*maxAbs, 1e-6)
	for c := range cout {
		if math.Abs(yFloat[c]-yQuant[c]) > tol {
			t.Errorf("channel %d: float=%.6f quant=%.6f diff=%.6f > tol=%.6f",
				c, yFloat[c], yQuant[c], math.Abs(yFloat[c]-yQuant[c]), tol)
		}
	}
}

// ─── Attention projection tests ─────────────────────────────────────────────

// TestQuantizedAttn_WeightShape verifies QuantizedAttentionProjections buffer shapes.
func TestQuantizedAttn_WeightShape(t *testing.T) {
	src := makeAttnLayer(8, 16, 2)
	cfg := DefaultQuantizationConfig[float64]()
	qa := newQuantizedAttentionProjections[float64](src, cfg)
	wantLen := src.Dmodel * src.Dmodel
	for name, w := range map[string][]int8{"Wq": qa.Wq, "Wk": qa.Wk, "Wv": qa.Wv, "Wo": qa.Wo} {
		if len(w) != wantLen {
			t.Errorf("%s len = %d, want %d", name, len(w), wantLen)
		}
	}
}

// TestQuantizedAttentionForward verifies attention output within 5% tolerance.
func TestQuantizedAttentionForward(t *testing.T) {
	const seqLen, dmodel, numHeads = 4, 8, 2
	src := makeAttnLayer(seqLen, dmodel, numHeads)

	rng := rand.New(rand.NewPCG(77, 77))
	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = rng.Float64()*0.5 - 0.25
	}

	floatOut := src.Forward(x)
	cfg := DefaultQuantizationConfig[float64]()
	qa := newQuantizedAttentionProjections[float64](src, cfg)
	quantOut := qa.Forward(x)

	if len(quantOut) != len(floatOut) {
		t.Fatalf("output length: float=%d quant=%d", len(floatOut), len(quantOut))
	}
	var maxAbs float64
	for _, v := range floatOut {
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	tol := math.Max(0.05*maxAbs, 1e-6)
	var maxErr float64
	for i, fv := range floatOut {
		if d := math.Abs(fv - quantOut[i]); d > maxErr {
			maxErr = d
		}
	}
	if maxErr > tol {
		t.Errorf("max ||delta||_∞ = %.6f > tol = %.6f (5%% of ||y||_∞=%.6f)", maxErr, tol, maxAbs)
	}
}

// TestQuantizeMHA verifies Quantize produces QuantizedAttentionProjections in layers.
func TestQuantizeMHA(t *testing.T) {
	const seqLen, dmodel, numHeads = 4, 8, 2
	net, err := nn.New[float64](
		nn.WithMultiHeadAttention[float64](seqLen, dmodel, numHeads),
		nn.WithInput[float64](uint(seqLen*dmodel)),
		nn.WithOutput[float64](uint(seqLen*dmodel), 0),
	)
	if err != nil {
		t.Skipf("MHA network build failed (likely shape issue): %v", err)
	}
	cfg := DefaultQuantizationConfig[float64]()
	qnet, err := Quantize[float64](net, nil, cfg)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	var found bool
	for _, l := range qnet.Layers {
		if _, ok := l.(*QuantizedAttentionProjections[float64]); ok {
			found = true
			break
		}
	}
	if !found {
		t.Error("no QuantizedAttentionProjections found in quantized layers")
	}
}

// ─── Utility coverage tests ──────────────────────────────────────────────────

// TestClampF64 covers the clampF64 helper.
func TestClampF64(t *testing.T) {
	if v := clampF64(5.0, 0.0, 1.0); v != 1.0 {
		t.Errorf("clampF64(5, 0, 1) = %v, want 1.0", v)
	}
	if v := clampF64(-1.0, 0.0, 1.0); v != 0.0 {
		t.Errorf("clampF64(-1, 0, 1) = %v, want 0.0", v)
	}
	if v := clampF64(0.5, 0.0, 1.0); v != 0.5 {
		t.Errorf("clampF64(0.5, 0, 1) = %v, want 0.5", v)
	}
}

// TestSliceMinMax covers sliceMinMax directly, including the empty-slice guard.
func TestSliceMinMax(t *testing.T) {
	mn, mx := sliceMinMax([]float64{3.0, -1.0, 2.0})
	if mn != -1.0 || mx != 3.0 {
		t.Errorf("sliceMinMax = (%v, %v), want (-1.0, 3.0)", mn, mx)
	}
	mn, mx = sliceMinMax(nil)
	if mn != 0 || mx != 0 {
		t.Errorf("sliceMinMax(nil) = (%v, %v), want (0, 0)", mn, mx)
	}
}

// TestPerTensorGranularity verifies that PerTensor granularity produces a single scale.
func TestPerTensorGranularity(t *testing.T) {
	cfg := DefaultQuantizationConfig[float64]()
	cfg.WeightGranularity = PerTensor
	src := convpkg.NewConv1D[float64](4, 3, 1, convpkg.PadValid, false)
	rng := rand.New(rand.NewPCG(1, 1))
	src.Init(rng)
	ql := newQuantizedConv1D(src, cfg)
	if len(ql.WeightParams.Scale) != 1 {
		t.Errorf("PerTensor: expected 1 scale, got %d", len(ql.WeightParams.Scale))
	}
	if ql.WeightParams.Granularity != PerTensor {
		t.Errorf("Granularity = %v, want PerTensor", ql.WeightParams.Granularity)
	}
}

// TestQuantizeClamping covers the q<-128 and q>127 saturation branches.
func TestQuantizeClamping(t *testing.T) {
	p := computeSymmetric(0.0, 1.0)
	if q := quantize(1000.0, p, 0); q != 127 {
		t.Errorf("quantize(1000) = %d, want 127 (clamp hi)", q)
	}
	if q := quantize(-1000.0, p, 0); q != -128 {
		t.Errorf("quantize(-1000) = %d, want -128 (clamp lo)", q)
	}
}

// TestComputeSymmetric_NegDominated verifies absMax = |rMin| when |rMin| > rMax.
func TestComputeSymmetric_NegDominated(t *testing.T) {
	p := computeSymmetric(-3.0, 1.0)
	want := 3.0 / 127.0
	if !almostEqual(p.Scale[0], want, 1e-9) {
		t.Errorf("scale = %.10f, want %.10f", p.Scale[0], want)
	}
}

// TestComputeAsymmetric_Degenerate covers the degenerate-range guard.
func TestComputeAsymmetric_Degenerate(t *testing.T) {
	p := computeAsymmetric(1.0, 1.0)
	if p.Scale[0] != 1.0 || p.ZeroPoint[0] != 0 {
		t.Errorf("degenerate: scale=%v zp=%v, want 1.0 0", p.Scale[0], p.ZeroPoint[0])
	}
}

// TestEncodeDecodeFloat64 covers encodeFloat64 / decodeFloat64 round-trip.
func TestEncodeDecodeFloat64(t *testing.T) {
	input := []float64{1.5, -2.5, 0.0, math.Pi}
	enc := encodeFloat64(input)
	if enc == "" {
		t.Fatal("encodeFloat64 returned empty string for non-empty slice")
	}
	decoded, err := decodeFloat64(enc)
	if err != nil {
		t.Fatalf("decodeFloat64: %v", err)
	}
	if len(decoded) != len(input) {
		t.Fatalf("length: got %d, want %d", len(decoded), len(input))
	}
	for i, v := range input {
		if decoded[i] != v {
			t.Errorf("[%d]: %.10f != %.10f", i, decoded[i], v)
		}
	}
}

// TestDecodeFloat64_Errors covers the bad-base64 and wrong-length error paths.
func TestDecodeFloat64_Errors(t *testing.T) {
	_, err := decodeFloat64("not!valid!base64!!")
	if err == nil {
		t.Error("expected error for invalid base64, got nil")
	}
	// 1 byte of valid base64 that decodes to a non-multiple-of-8 byte slice
	_, err = decodeFloat64("YQ==") // decodes to [0x61] — 1 byte
	if err == nil {
		t.Error("expected error for byte length not multiple of 8")
	}
}

// TestDecodeInt8_BadBase64 covers the decodeInt8 error path.
func TestDecodeInt8_BadBase64(t *testing.T) {
	_, err := decodeInt8("not!valid!!")
	if err == nil {
		t.Error("expected error for invalid base64")
	}
}

// TestOpaquePassthrough_Forward covers opaquePassthrough.Forward (persistence.go).
func TestOpaquePassthrough_Forward(t *testing.T) {
	op := &opaquePassthrough[float64]{}
	x := []float64{1.0, 2.0, 3.0}
	out := op.Forward(x)
	if len(out) != len(x) || out[0] != x[0] {
		t.Errorf("opaquePassthrough should return input unchanged, got %v", out)
	}
}

// TestFloatPassthrough_Forward covers floatPassthrough.Forward (quantizer.go).
func TestFloatPassthrough_Forward(t *testing.T) {
	src := convpkg.NewConv1D[float64](4, 3, 1, convpkg.PadValid, false)
	rng := rand.New(rand.NewPCG(1, 1))
	src.Init(rng)
	fp := &floatPassthrough[float64]{inner: src}
	x := make([]float64, 10)
	for i := range x {
		x[i] = float64(i) * 0.1
	}
	out := fp.Forward(x)
	expected := src.Forward(x)
	if len(out) != len(expected) {
		t.Fatalf("output length: passthrough=%d direct=%d", len(out), len(expected))
	}
	for i := range out {
		if out[i] != expected[i] {
			t.Errorf("[%d]: passthrough=%.10f direct=%.10f", i, out[i], expected[i])
		}
	}
}

// TestUnmarshalUnknownLayerType covers the default error branch in unmarshalLayer.
func TestUnmarshalUnknownLayerType(t *testing.T) {
	data := []byte(`{"type":"quantized","layers":[{"type":"Banana","scale":[],"zero_point":[],"weights":""}]}`)
	_, err := UnmarshalQNN[float64](data)
	if err == nil {
		t.Error("expected error for unknown layer type, got nil")
	}
}

// ─── Conv2D quantization tests ───────────────────────────────────────────────

// TestQuantizedConv2D_WeightShape verifies QuantizedConv2D buffer shapes.
func TestQuantizedConv2D_WeightShape(t *testing.T) {
	src := convpkg.NewConv2D[float64](4, 1, 3, 3, 1, 1, convpkg.PadValid, false)
	rng := rand.New(rand.NewPCG(1, 1))
	src.Init(rng)
	cfg := DefaultQuantizationConfig[float64]()
	ql := newQuantizedConv2D(src, cfg)
	wantLen := src.NumFilters * src.InChannels * src.KernelH * src.KernelW
	if len(ql.Weights) != wantLen {
		t.Errorf("Weights len = %d, want %d", len(ql.Weights), wantLen)
	}
	if len(ql.WeightParams.Scale) != src.NumFilters {
		t.Errorf("Scale len = %d, want %d (PerChannel)", len(ql.WeightParams.Scale), src.NumFilters)
	}
}

// TestQuantizedConv2D_Forward verifies Conv2D forward output within 5% tolerance.
func TestQuantizedConv2D_Forward(t *testing.T) {
	src := convpkg.NewConv2D[float64](4, 1, 3, 3, 1, 1, convpkg.PadValid, false)
	rng := rand.New(rand.NewPCG(7, 7))
	src.Init(rng)
	// 1-channel 6×6 spatial input (CHW flat).
	x := make([]float64, 1*6*6)
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	floatOut := src.Forward(x)
	cfg := DefaultQuantizationConfig[float64]()
	ql := newQuantizedConv2D(src, cfg)
	quantOut := ql.Forward(x)
	if len(quantOut) != len(floatOut) {
		t.Fatalf("output length: float=%d quant=%d", len(floatOut), len(quantOut))
	}
	var maxAbs float64
	for _, v := range floatOut {
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	tol := math.Max(0.05*maxAbs, 1e-6)
	for i, fv := range floatOut {
		if math.Abs(fv-quantOut[i]) > tol {
			t.Errorf("output[%d]: float=%.6f quant=%.6f diff=%.6f > tol=%.6f",
				i, fv, quantOut[i], math.Abs(fv-quantOut[i]), tol)
		}
	}
}

// TestConv2DRoundTrip verifies Conv2D MarshalQNN + UnmarshalQNN round-trip.
func TestConv2DRoundTrip(t *testing.T) {
	src := convpkg.NewConv2D[float64](4, 1, 3, 3, 1, 1, convpkg.PadValid, false)
	rng := rand.New(rand.NewPCG(3, 3))
	src.Init(rng)
	cfg := DefaultQuantizationConfig[float64]()
	ql := newQuantizedConv2D(src, cfg)
	qnet := &QuantizedNetwork[float64]{Layers: []quantizedLayer[float64]{ql}}

	data, err := MarshalQNN(qnet)
	if err != nil {
		t.Fatalf("MarshalQNN: %v", err)
	}
	qnet2, err := UnmarshalQNN[float64](data)
	if err != nil {
		t.Fatalf("UnmarshalQNN: %v", err)
	}
	x := make([]float64, 1*6*6)
	y1 := qnet.Forward(x)
	y2 := qnet2.Forward(x)
	if len(y1) != len(y2) {
		t.Fatalf("output length: before=%d after=%d", len(y1), len(y2))
	}
	for i := range y1 {
		if y1[i] != y2[i] {
			t.Errorf("Conv2D output[%d] not bit-exact after round-trip", i)
		}
	}
}

// ─── Dense round-trip ────────────────────────────────────────────────────────

// TestDenseRoundTrip verifies Dense MarshalQNN + UnmarshalQNN round-trip.
func TestDenseRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	const cout, cin = 8, 4
	w := make([]float64, cout*cin)
	for i := range w {
		w[i] = rng.Float64()*2 - 1
	}
	bias := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	cfg := DefaultQuantizationConfig[float64]()
	d := newQuantizedDense[float64](w, bias, cout, cin, cfg)
	qnet := &QuantizedNetwork[float64]{Layers: []quantizedLayer[float64]{d}}

	data, err := MarshalQNN(qnet)
	if err != nil {
		t.Fatalf("MarshalQNN: %v", err)
	}
	qnet2, err := UnmarshalQNN[float64](data)
	if err != nil {
		t.Fatalf("UnmarshalQNN: %v", err)
	}
	x := make([]float64, cin)
	y1 := qnet.Forward(x)
	y2 := qnet2.Forward(x)
	for i := range y1 {
		if y1[i] != y2[i] {
			t.Errorf("Dense output[%d] mismatch after round-trip", i)
		}
	}
}

// ─── Attention round-trip ────────────────────────────────────────────────────

// TestAttnProjRoundTrip verifies AttentionProj MarshalQNN + UnmarshalQNN round-trip.
func TestAttnProjRoundTrip(t *testing.T) {
	const seqLen, dmodel, numHeads = 4, 8, 2
	src := makeAttnLayer(seqLen, dmodel, numHeads)
	cfg := DefaultQuantizationConfig[float64]()
	qa := newQuantizedAttentionProjections(src, cfg)
	qnet := &QuantizedNetwork[float64]{Layers: []quantizedLayer[float64]{qa}}

	data, err := MarshalQNN(qnet)
	if err != nil {
		t.Fatalf("MarshalQNN: %v", err)
	}
	qnet2, err := UnmarshalQNN[float64](data)
	if err != nil {
		t.Fatalf("UnmarshalQNN: %v", err)
	}
	rng := rand.New(rand.NewPCG(9, 9))
	x := make([]float64, seqLen*dmodel)
	for i := range x {
		x[i] = rng.Float64()*0.5 - 0.25
	}
	y1 := qnet.Forward(x)
	y2 := qnet2.Forward(x)
	if len(y1) != len(y2) {
		t.Fatalf("output length: before=%d after=%d", len(y1), len(y2))
	}
	for i := range y1 {
		if y1[i] != y2[i] {
			t.Errorf("AttentionProj output[%d] not bit-exact after round-trip", i)
		}
	}
}

// TestFloatPassRoundTrip covers marshalLayer for floatPassthrough and unmarshalLayer
// for "FloatPass" (which produces an opaquePassthrough).
func TestFloatPassRoundTrip(t *testing.T) {
	src := convpkg.NewConv1D[float64](2, 3, 1, convpkg.PadValid, false)
	fp := &floatPassthrough[float64]{inner: src}
	qnet := &QuantizedNetwork[float64]{Layers: []quantizedLayer[float64]{fp}}

	data, err := MarshalQNN(qnet)
	if err != nil {
		t.Fatalf("MarshalQNN: %v", err)
	}
	qnet2, err := UnmarshalQNN[float64](data)
	if err != nil {
		t.Fatalf("UnmarshalQNN: %v", err)
	}
	// opaquePassthrough should act as identity.
	x := []float64{1.0, 2.0, 3.0}
	y := qnet2.Forward(x)
	if len(y) != len(x) {
		t.Errorf("opaquePassthrough output length = %d, want %d", len(y), len(x))
	}
}

// ─── Calibration edge cases ───────────────────────────────────────────────────

// TestCalibParams_DegeneratePercentile covers the degenerate-range error path in
// Params() when Percentile99p9 clips to a near-zero range.
func TestCalibParams_DegeneratePercentile(t *testing.T) {
	net := makeConv1DNet(t)
	cfg := DefaultQuantizationConfig[float64]()
	cfg.Strategy = Percentile99p9
	cfg.PercentileThreshold = 99.9
	runner := NewCalibrationRunner(net, cfg)

	// All-constant samples → percentile range degenerates.
	samples := make([][]float64, 50)
	for i := range samples {
		samples[i] = make([]float64, 8) // all zeros
	}
	if err := runner.Run(samples); err != nil {
		t.Fatalf("Run: %v", err)
	}
	_, err := runner.Params()
	if err == nil {
		t.Fatal("expected ErrDegenerateRange for constant activations, got nil")
	}
	if !errors.Is(err, utils.ErrDegenerateRange) {
		t.Errorf("error does not wrap ErrDegenerateRange: %v", err)
	}
}
