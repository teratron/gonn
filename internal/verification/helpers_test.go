// Package verification is the numeric safety net for the restoration work.
//
// Every test here encodes the mathematically CORRECT target behavior, not
// the current implementation. Tests that fail against unfixed code are the
// point: they gate each restoration phase.
//
// Phase 0 note: these helpers are written against the CURRENT public API so
// the suite compiles and runs RED today (assertion failures that pinpoint the
// bugs), rather than failing to build. Signatures that later phases change
// (e.g. ApplyFlatWeights gaining an error return in Phase 3) are updated here
// when that phase lands.
package verification

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

// netSpec describes one dense topology under test.
type netSpec struct {
	name    string
	hidden  []uint
	act     activation.Type
	outAct  activation.Type
	outputs uint
	lossT   loss.Type
}

// buildDense compiles a deterministic dense network for spec.
func buildDense(t *testing.T, spec netSpec, inputs uint, extra ...nn.Option[float64]) *nn.NN[float64] {
	t.Helper()
	opts := []nn.Option[float64]{
		nn.WithInput[float64](inputs),
		nn.WithBias[float64](true),
		nn.WithOutput[float64](spec.outputs, spec.outAct),
		nn.WithLoss[float64](spec.lossT),
		nn.WithLearningRate[float64](0.1),
		nn.WithMaxIterations[float64](1),
		nn.WithWeightInitSeed[float64](12345),
	}
	for _, h := range spec.hidden {
		opts = append(opts, nn.WithHiddenLayer[float64](h, spec.act))
	}
	opts = append(opts, extra...)
	n, err := nn.New(opts...)
	if err != nil {
		t.Fatalf("build %s: %v", spec.name, err)
	}
	return n
}

// refLossScalar computes the reference scalar loss Σ_i loss.Loss(y_i, t_i, mode)
// from a forward pass. It matches the gradient convention AppendFlatGradients
// must satisfy: grad = ∂(Σ loss.Loss(y_i, t_i))/∂w.
//
// Restricted to element-wise losses for Phase 0. Vector losses (CCE etc.) get
// their own oracle once loss.VectorDerivative lands in Phase 1.
func refLossScalar(t *testing.T, n *nn.NN[float64], input, target []float64, mode loss.Type) float64 {
	t.Helper()
	if err := n.SetInputs(input); err != nil {
		t.Fatalf("SetInputs: %v", err)
	}
	if err := n.SetTargets(target); err != nil {
		t.Fatalf("SetTargets: %v", err)
	}
	n.CalculateValues()
	var s float64
	for i, c := range n.Network.Output.Cells() {
		s += loss.Loss(*c.GetValue(), target[i], mode)
	}
	return s
}

// gradcheckDense compares AppendFlatGradients against central differences of
// refLossScalar for every weight. Returns the worst relative error.
func gradcheckDense(t *testing.T, n *nn.NN[float64], input, target []float64, mode loss.Type) float64 {
	t.Helper()
	if err := n.SetInputs(input); err != nil {
		t.Fatalf("SetInputs: %v", err)
	}
	if err := n.SetTargets(target); err != nil {
		t.Fatalf("SetTargets: %v", err)
	}
	n.CalculateValues()
	n.CalculateMisses()
	ana := n.AppendFlatGradients(nil)

	w0 := n.FlatWeights()
	if len(ana) != len(w0) {
		t.Fatalf("gradient/weight length mismatch: %d vs %d", len(ana), len(w0))
	}
	const h = 1e-6
	worst := 0.0
	w := make([]float64, len(w0))
	for i := range w0 {
		copy(w, w0)
		w[i] = w0[i] + h
		if err := n.ApplyFlatWeights(w); err != nil {
			t.Fatalf("ApplyFlatWeights(+h): %v", err)
		}
		lp := refLossScalar(t, n, input, target, mode)
		w[i] = w0[i] - h
		if err := n.ApplyFlatWeights(w); err != nil {
			t.Fatalf("ApplyFlatWeights(-h): %v", err)
		}
		lm := refLossScalar(t, n, input, target, mode)
		num := (lp - lm) / (2 * h)
		scale := math.Abs(num) + math.Abs(ana[i])
		if scale < 1e-10 {
			continue // both effectively zero — no signal
		}
		rel := math.Abs(num-ana[i]) / scale
		if rel > worst {
			worst = rel
		}
	}
	if err := n.ApplyFlatWeights(w0); err != nil {
		t.Fatalf("ApplyFlatWeights(restore): %v", err)
	}
	return worst
}

func absDiffMax(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		d := math.Abs(a[i] - b[i])
		if d > m {
			m = d
		}
	}
	return m
}

func xorSamples() []nn.Sample[float64] {
	return []nn.Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
}
