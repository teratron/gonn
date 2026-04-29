package network

import (
	"errors"
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

func newTrainable[T utils.Float](inSize, hiddenSize, outSize int, withBias bool) (*Network[T], error) {
	in := layer.NewInput[T](inSize)
	hidden := layer.NewDense[T](hiddenSize, activation.SIGMOID, withBias)
	out := layer.NewOutput[T](outSize, activation.SIGMOID, loss.MSE, withBias)
	n := New[T]()
	if err := n.SetLayers(in, hidden, out); err != nil {
		return nil, err
	}
	if err := n.Build(); err != nil {
		return nil, err
	}
	return &n, nil
}

func TestNewSetsDefaults(t *testing.T) {
	t.Parallel()
	n := New[float64]()
	if n.LearningRate != 0.3 {
		t.Errorf("LearningRate = %v; want 0.3", n.LearningRate)
	}
}

func TestSetLayersRejectsNil(t *testing.T) {
	t.Parallel()
	n := New[float64]()
	err := n.SetLayers(nil, nil, nil)
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
}

func TestSetLayersRejectsZeroSize(t *testing.T) {
	t.Parallel()
	in := layer.NewInput[float64](0)
	hidden := layer.NewDense[float64](2, activation.SIGMOID, false)
	out := layer.NewOutput[float64](1, activation.SIGMOID, loss.MSE, false)
	n := New[float64]()
	err := n.SetLayers(in, hidden, out)
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
}

func TestBuildRejectsEmptyBundle(t *testing.T) {
	t.Parallel()
	n := New[float64]()
	err := n.Build()
	if !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("expected ErrUserConfig, got %v", err)
	}
}

func TestBuildWiresAxons(t *testing.T) {
	t.Parallel()
	n, err := newTrainable[float64](2, 3, 1, true)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	for i, h := range n.Hidden.Cells() {
		// 2 inputs + 1 bias = 3 axons per hidden cell
		if got := len(h.Axons); got != 3 {
			t.Errorf("hidden[%d] axons = %d; want 3", i, got)
		}
	}
	for i, o := range n.Output.Cells() {
		// 3 hidden + 1 bias = 4 axons per output cell
		if got := len(o.Axons); got != 4 {
			t.Errorf("output[%d] axons = %d; want 4", i, got)
		}
	}
}

func TestSetInputsRejectsLengthMismatch(t *testing.T) {
	t.Parallel()
	n, _ := newTrainable[float64](2, 2, 1, false)
	err := n.SetInputs([]float64{1.0})
	if !errors.Is(err, utils.ErrInputData) {
		t.Errorf("expected ErrInputData, got %v", err)
	}
}

func TestSetTargetsRejectsLengthMismatch(t *testing.T) {
	t.Parallel()
	n, _ := newTrainable[float64](2, 2, 1, false)
	err := n.SetTargets([]float64{1.0, 2.0})
	if !errors.Is(err, utils.ErrInputData) {
		t.Errorf("expected ErrInputData, got %v", err)
	}
}

func TestSetInputsWritesEachCell(t *testing.T) {
	t.Parallel()
	// Regression of [l2-network-graph] §5.4 #4: legacy code wrote every
	// data point into cells[0] in a loop. The rewrite must place each
	// value in its matching cell.
	n, _ := newTrainable[float64](3, 2, 1, false)
	if err := n.SetInputs([]float64{0.1, 0.2, 0.3}); err != nil {
		t.Fatalf("SetInputs: %v", err)
	}
	for i, want := range []float64{0.1, 0.2, 0.3} {
		if got := *n.Input.Cells()[i].GetValue(); got != want {
			t.Errorf("input[%d] = %v; want %v", i, got, want)
		}
	}
}

// TestXORConvergence — the canonical Phase-1 smoke test. A 2-3-1 network
// with sigmoid activations must drive MSE below 0.05 on the XOR task
// within a generous epoch budget. Failure indicates either a propagation
// bug or a regression in the activation / weight-update pipeline.
func TestXORConvergence(t *testing.T) {
	t.Parallel()
	n, err := newTrainable[float64](2, 4, 1, true)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	// Higher rate accelerates convergence on this tiny problem.
	n.LearningRate = 0.5

	dataset := [][2][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	const maxEpochs = 5000
	var lastLoss float64
	for epoch := 0; epoch < maxEpochs; epoch++ {
		var total float64
		for _, sample := range dataset {
			l, err := n.Train(sample[0], sample[1])
			if err != nil {
				t.Fatalf("Train epoch=%d: %v", epoch, err)
			}
			total += float64(l)
		}
		lastLoss = total / float64(len(dataset))
		if lastLoss < 0.05 {
			t.Logf("XOR converged in %d epochs, mean loss = %v", epoch+1, lastLoss)
			break
		}
	}
	if lastLoss >= 0.05 {
		t.Errorf("XOR did not converge: mean loss after %d epochs = %v (want < 0.05)", maxEpochs, lastLoss)
	}

	// Sanity: after training, predictions must reflect XOR.
	for _, sample := range dataset {
		if err := n.SetInputs(sample[0]); err != nil {
			t.Fatalf("SetInputs: %v", err)
		}
		n.CalculateValues()
		got := *n.Output.Cells()[0].GetValue()
		want := sample[1][0]
		if math.Abs(float64(got)-want) > 0.3 {
			t.Errorf("XOR(%v) = %v; want ~%v", sample[0], got, want)
		}
	}
}

func TestCalculateLossDefaultUsesConfiguredMode(t *testing.T) {
	t.Parallel()
	n, _ := newTrainable[float64](1, 1, 1, false)
	if got := n.LossMode(); got != loss.MSE {
		t.Errorf("LossMode = %v; want MSE", got)
	}
	if err := n.SetInputs([]float64{1.0}); err != nil {
		t.Fatal(err)
	}
	if err := n.SetTargets([]float64{0.5}); err != nil {
		t.Fatal(err)
	}
	n.CalculateValues()
	def := n.CalculateLossDefault()
	explicit := n.CalculateLoss(loss.MSE)
	if float64(def) != float64(explicit) {
		t.Errorf("default loss %v != explicit MSE loss %v", def, explicit)
	}
}

func TestBundleAddAt(t *testing.T) {
	t.Parallel()
	n, _ := newTrainable[float64](2, 2, 1, false)
	if got := n.Hidden.Len(); got != 2 {
		t.Errorf("Hidden.Len = %d; want 2", got)
	}
	if n.Hidden.At(0) == nil {
		t.Errorf("Hidden.At(0) returned nil")
	}
}
