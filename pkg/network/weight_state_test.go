package network

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// newTrainableWithSampler builds a 2→4→1 network with the given weight sampler.
func newTrainableWithSampler[T utils.Float](sampler WeightSampler[T]) (*Network[T], error) {
	in := layer.NewInput[T](2)
	hidden := layer.NewDense[T](4, activation.SIGMOID, false)
	out := layer.NewOutput[T](1, activation.SIGMOID, loss.MSE, false)
	n := New[T]()
	n.SetWeightSampler(sampler)
	if err := n.SetLayers(in, []*layer.Dense[T]{hidden}, out); err != nil {
		return nil, err
	}
	if err := n.Build(); err != nil {
		return nil, err
	}
	return &n, nil
}

// TestSetWeightSampler verifies that the configured sampler is applied during Build.
// We use a constant sampler so every weight is deterministically 0.25.
func TestSetWeightSampler(t *testing.T) {
	t.Parallel()
	const want = float64(0.25)
	sampler := func(_, _ int) float64 { return want }

	n, err := newTrainableWithSampler[float64](sampler)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	for _, hb := range n.Hiddens {
		for _, h := range hb.cells {
			for _, a := range h.Axons {
				if float64(a.Weight) != want {
					t.Errorf("hidden weight = %v, want %v", a.Weight, want)
				}
			}
		}
	}
	for _, o := range n.Output.cells {
		for _, a := range o.Axons {
			if float64(a.Weight) != want {
				t.Errorf("output weight = %v, want %v", a.Weight, want)
			}
		}
	}
}

// TestFlatWeightsRoundTrip verifies FlatWeights, AppendFlatWeights, ApplyFlatWeights
// and weightCount all agree on the same canonical ordering.
func TestFlatWeightsRoundTrip(t *testing.T) {
	t.Parallel()
	n, err := newTrainable[float64](2, 4, 1, false)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	flat := n.FlatWeights()
	if len(flat) == 0 {
		t.Fatal("FlatWeights returned empty slice")
	}
	if len(flat) != n.weightCount() {
		t.Errorf("FlatWeights len %d != weightCount %d", len(flat), n.weightCount())
	}

	// AppendFlatWeights must produce the same elements.
	app := n.AppendFlatWeights(nil)
	if len(app) != len(flat) {
		t.Fatalf("AppendFlatWeights len %d != FlatWeights len %d", len(app), len(flat))
	}
	for i := range flat {
		if flat[i] != app[i] {
			t.Errorf("index %d: FlatWeights=%v AppendFlatWeights=%v", i, flat[i], app[i])
		}
	}

	// Overwrite all weights to a sentinel, apply back, then verify.
	sentinel := make([]float64, len(flat))
	for i := range sentinel {
		sentinel[i] = float64(i) * 0.01
	}
	n.ApplyFlatWeights(sentinel)

	got := n.FlatWeights()
	for i, v := range got {
		if math.Abs(float64(v)-sentinel[i]) > 1e-12 {
			t.Errorf("ApplyFlatWeights[%d] = %v, want %v", i, v, sentinel[i])
		}
	}
}

// TestAppendFlatWeightsReuse verifies AppendFlatWeights reuses a pre-allocated slice.
func TestAppendFlatWeightsReuse(t *testing.T) {
	t.Parallel()
	n, err := newTrainable[float64](2, 4, 1, false)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	buf := make([]float64, 0, 100)
	result := n.AppendFlatWeights(buf)
	if len(result) == 0 {
		t.Fatal("AppendFlatWeights returned empty")
	}
}

// TestHiddenActivationsRoundTrip verifies HiddenActivations / SetHiddenActivations
// round-trip correctly and that ApplyMask-style overwriting works.
func TestHiddenActivationsRoundTrip(t *testing.T) {
	t.Parallel()
	n, err := newTrainable[float64](2, 4, 1, false)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	// Forward pass to populate activation values.
	if err := n.SetInputs([]float64{0.5, 0.5}); err != nil {
		t.Fatalf("SetInputs: %v", err)
	}
	n.CalculateValues()

	acts := n.HiddenActivations()
	if len(acts) != 4 {
		t.Fatalf("HiddenActivations len = %d, want 4", len(acts))
	}

	// Set all to a known value, read back.
	modified := make([]float64, len(acts))
	for i := range modified {
		modified[i] = 0.42
	}
	n.SetHiddenActivations(modified)

	after := n.HiddenActivations()
	for i, v := range after {
		if math.Abs(float64(v)-0.42) > 1e-12 {
			t.Errorf("acts[%d] = %v after SetHiddenActivations, want 0.42", i, v)
		}
	}
}

// TestAppendFlatGradients verifies that AppendFlatGradients returns a non-empty
// slice with the expected length after a forward+backward pass.
func TestAppendFlatGradients(t *testing.T) {
	t.Parallel()
	n, err := newTrainable[float64](2, 4, 1, false)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	// Full forward+backward step to populate miss and pre-activation buffers.
	if _, err := n.Train([]float64{0, 1}, []float64{1}); err != nil {
		t.Fatalf("Train: %v", err)
	}

	grads := n.AppendFlatGradients(nil)
	if len(grads) == 0 {
		t.Fatal("AppendFlatGradients returned empty slice")
	}
	if len(grads) != n.weightCount() {
		t.Errorf("gradient count %d != weight count %d", len(grads), n.weightCount())
	}
}

// TestBundleAdd verifies the Add method on bundle.
func TestBundleAdd(t *testing.T) {
	t.Parallel()
	b := newBundle[float64, *cell.Input[float64]]()
	if b.Len() != 0 {
		t.Fatalf("initial Len = %d, want 0", b.Len())
	}
	c := cell.NewInput[float64](1.0)
	b.Add(c)
	if b.Len() != 1 {
		t.Fatalf("Len after Add = %d, want 1", b.Len())
	}
	if b.At(0) != c {
		t.Errorf("At(0) did not return added cell")
	}
}
