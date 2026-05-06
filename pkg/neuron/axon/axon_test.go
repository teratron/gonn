package axon

import (
	"github.com/teratron/gonn/pkg/utils"
	"math"
	"sync"
	"testing"

	"github.com/teratron/gonn/pkg/neuron"
)

// stubNucleus is a minimal Nucleus[T] used as the incoming endpoint —
// avoids the cell-package import cycle (cell already depends on axon).
type stubNucleus[T neuronFloat] struct{ v T }

func (s *stubNucleus[T]) GetValue() *T { return &s.v }

// stubNeuron is a minimal Neuron[T] used as the outgoing endpoint.
type stubNeuron[T neuronFloat] struct {
	v, miss T
}

func (s *stubNeuron[T]) GetValue() *T         { return &s.v }
func (s *stubNeuron[T]) GetMiss() *T          { return &s.miss }
func (s *stubNeuron[T]) SetMiss(value T)      { s.miss = value }
func (s *stubNeuron[T]) CalculateValue()      {}
func (s *stubNeuron[T]) CalculateWeight(_ *T) {}

// neuronFloat is a private alias of utils.Float to keep stub signatures
// concise — the public utils.Float constraint is what production code
// uses, this is only for stub plumbing inside the test file.
type neuronFloat interface {
	utils.Float
}

func approx(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

func TestNewAssignsBothEndpoints(t *testing.T) {
	t.Parallel()
	in := &stubNucleus[float64]{v: 0.5}
	out := &stubNeuron[float64]{}
	a := New[float64](in, out)
	if a.Cell != neuron.Nucleus[float64](in) {
		t.Errorf("New did not record the incoming endpoint")
	}
	if a.OutgoingCell != neuron.Neuron[float64](out) {
		t.Errorf("New did not record the outgoing endpoint (regression of C-001)")
	}
}

func TestNewDefaultWeightInBaselineRange(t *testing.T) {
	t.Parallel()
	in := &stubNucleus[float64]{}
	out := &stubNeuron[float64]{}
	for range 1024 {
		a := New[float64](in, out)
		if a.Weight < -0.5 || a.Weight > 0.5 {
			t.Fatalf("default weight %v outside legacy [-0.5, 0.5] baseline", a.Weight)
		}
	}
}

func TestNewWithWeightHonoursCallerValue(t *testing.T) {
	t.Parallel()
	in := &stubNucleus[float64]{v: 1.0}
	out := &stubNeuron[float64]{}
	a := NewWithWeight[float64](0.42, in, out)
	if a.Weight != 0.42 {
		t.Errorf("NewWithWeight stored %v; want 0.42", a.Weight)
	}
}

func TestCalculateValue(t *testing.T) {
	t.Parallel()
	in := &stubNucleus[float64]{v: 3.0}
	out := &stubNeuron[float64]{}
	a := NewWithWeight[float64](0.5, in, out)
	if got := a.CalculateValue(); !approx(got, 1.5) {
		t.Errorf("CalculateValue = %v; want 1.5 (3.0 * 0.5)", got)
	}
}

func TestCalculateMissReadsOutgoing(t *testing.T) {
	t.Parallel()
	in := &stubNucleus[float64]{v: 0}
	out := &stubNeuron[float64]{miss: 4.0}
	a := NewWithWeight[float64](0.25, in, out)
	if got := a.CalculateMiss(); !approx(got, 1.0) {
		t.Errorf("CalculateMiss = %v; want 1.0 (4 * 0.25)", got)
	}
}

func TestCalculateWeightUpdatesInPlace(t *testing.T) {
	t.Parallel()
	in := &stubNucleus[float64]{v: 2.0}
	out := &stubNeuron[float64]{}
	a := NewWithWeight[float64](1.0, in, out)
	g := 0.1
	a.CalculateWeight(&g)
	if !approx(float64(a.Weight), 1.2) {
		t.Errorf("weight after gradient step = %v; want 1.2 (1.0 + 0.1 * 2.0)", a.Weight)
	}
}

// TestNewIsConcurrencySafe runs many goroutines through the default-weight
// path. The package mutex guarding the shared PCG must serialise access —
// failure would surface as a data race or a panic from rand internals.
func TestNewIsConcurrencySafe(t *testing.T) {
	t.Parallel()
	const goroutines, perG = 16, 64
	in := &stubNucleus[float64]{}
	out := &stubNeuron[float64]{}
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for range goroutines {
		go func() {
			defer wg.Done()
			for range perG {
				a := New[float64](in, out)
				if a.Weight < -0.5 || a.Weight > 0.5 {
					t.Errorf("concurrent New produced out-of-range weight %v", a.Weight)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestFloat32Path(t *testing.T) {
	t.Parallel()
	in := &stubNucleus[float32]{v: 1.5}
	out := &stubNeuron[float32]{}
	a := NewWithWeight[float32](2.0, in, out)
	if got := a.CalculateValue(); float64(got) != 3.0 {
		t.Errorf("float32 CalculateValue = %v; want 3.0", got)
	}
}
