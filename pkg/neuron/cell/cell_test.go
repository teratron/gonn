package cell

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/axon"
)

// approxEqual returns true if |a - b| <= 1e-9 — sufficient for the
// integer-derived sums that appear in these tests.
func approxEqual(a, b float64) bool {
	return math.Abs(a-b) <= 1e-9
}

func TestCoreGetSetValue(t *testing.T) {
	t.Parallel()
	c := newCore[float64]([2]uint{1, 2})
	if c.Id != [2]uint{1, 2} {
		t.Fatalf("Id = %v; want {1, 2}", c.Id)
	}
	if got := *c.GetValue(); got != 0 {
		t.Errorf("freshly built core value = %v; want 0", got)
	}
	c.SetValue(3.5)
	if got := *c.GetValue(); !approxEqual(got, 3.5) {
		t.Errorf("after SetValue(3.5), GetValue = %v; want 3.5", got)
	}
}

func TestInputConstructorAndIO(t *testing.T) {
	t.Parallel()
	in := NewInput[float32](2.0)
	if in.Id[0] != uint(neuron.INPUT) {
		t.Errorf("Input.Id[0] = %d; want %d", in.Id[0], neuron.INPUT)
	}
	if got := *in.GetValue(); got != 2.0 {
		t.Errorf("GetValue = %v; want 2.0", got)
	}
	in.SetValue(7.5)
	if got := *in.GetValue(); got != 7.5 {
		t.Errorf("after SetValue, GetValue = %v; want 7.5", got)
	}
}

func TestBiasIsConstantOne(t *testing.T) {
	t.Parallel()
	b := NewBias[float64]()
	if b.Id[0] != uint(neuron.BIAS) {
		t.Errorf("Bias.Id[0] = %d; want %d", b.Id[0], neuron.BIAS)
	}
	if got := *b.GetValue(); got != 1.0 {
		t.Errorf("Bias.GetValue = %v; want 1.0", got)
	}
}

func TestDenseMissAccessors(t *testing.T) {
	t.Parallel()
	d := NewDense[float64](0)
	if got := *d.GetMiss(); got != 0 {
		t.Errorf("fresh GetMiss = %v; want 0", got)
	}
	d.SetMiss(2.0)
	if got := *d.GetMiss(); got != 2.0 {
		t.Errorf("after SetMiss(2), GetMiss = %v; want 2", got)
	}
	d.AddMiss(0.5)
	d.AddMiss(0.5)
	if got := *d.GetMiss(); !approxEqual(got, 3.0) {
		t.Errorf("after two AddMiss(0.5), GetMiss = %v; want 3", got)
	}
}

func TestDenseCalculateValueSumsAxons(t *testing.T) {
	t.Parallel()
	in1 := NewInput[float64](2.0)
	in2 := NewInput[float64](3.0)
	d := NewDense[float64](0)
	d.Axons = append(d.Axons,
		axon.NewWithWeight[float64](0.5, in1, d),
		axon.NewWithWeight[float64](-0.25, in2, d),
	)
	d.CalculateValue()
	want := 2.0*0.5 + 3.0*(-0.25)
	if got := *d.GetValue(); !approxEqual(got, want) {
		t.Errorf("CalculateValue = %v; want %v", got, want)
	}
}

func TestDenseCalculateWeightAppliesGradient(t *testing.T) {
	t.Parallel()
	in1 := NewInput[float64](2.0)
	d := NewDense[float64](0)
	d.Axons = append(d.Axons, axon.NewWithWeight[float64](1.0, in1, d))
	d.SetMiss(0.5)
	rate := 0.1
	d.CalculateWeight(&rate)
	// gradient = rate * miss = 0.05
	// new weight = 1.0 + gradient * cell.value = 1.0 + 0.05 * 2.0 = 1.1
	if got := d.Axons[0].W(); !approxEqual(got, 1.1) {
		t.Errorf("axon.W() after backward = %v; want 1.1", got)
	}
}

func TestOutputCalculateValueDoesNotRecurse(t *testing.T) {
	t.Parallel()
	in := NewInput[float64](2.0)
	target := 7.0
	o := NewOutput[float64](&target)
	o.Axons = append(o.Axons, axon.NewWithWeight[float64](1.5, in, o))
	// If the legacy recursion were still present, this call would either
	// stack-overflow or hang. Done in the same goroutine — a panic surfaces
	// as a test failure via the deferred recover below.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Output.CalculateValue panicked (likely recursion): %v", r)
		}
	}()
	o.CalculateValue()
	if got := *o.GetValue(); !approxEqual(got, 3.0) {
		t.Errorf("Output value = %v; want 3.0", got)
	}
	// miss = target - value = 7 - 3 = 4
	if got := *o.GetMiss(); !approxEqual(got, 4.0) {
		t.Errorf("Output miss = %v; want 4.0", got)
	}
}

func TestOutputTargetAccessors(t *testing.T) {
	t.Parallel()
	target := 0.5
	o := NewOutput[float64](&target)
	if o.GetTarget() != &target {
		t.Errorf("GetTarget did not return the supplied pointer")
	}
	other := 1.0
	o.SetTarget(&other)
	if o.GetTarget() != &other {
		t.Errorf("after SetTarget, GetTarget did not return the new pointer")
	}
}

func TestOutputCalculateValueWithNilTarget(t *testing.T) {
	t.Parallel()
	in := NewInput[float64](1.0)
	o := NewOutput[float64](nil)
	o.Axons = append(o.Axons, axon.NewWithWeight[float64](2.0, in, o))
	o.CalculateValue()
	// Forward must still run; miss stays at zero because no target is set.
	if got := *o.GetValue(); !approxEqual(got, 2.0) {
		t.Errorf("nil-target value = %v; want 2.0", got)
	}
	if got := *o.GetMiss(); got != 0 {
		t.Errorf("nil-target miss = %v; want 0 (no residual without target)", got)
	}
}

func TestHiddenIsDenseAlias(t *testing.T) {
	t.Parallel()
	h := NewHidden[float64](3)
	// Generic alias: Hidden[T] is literally Dense[T]. The interface
	// assertion in hidden.go would already fail at compile time if this
	// were broken; the runtime check just guards regression.
	var _ neuron.Neuron[float64] = h
	if h.Id[0] != uint(neuron.DENSE) {
		t.Errorf("Hidden.Id[0] = %d; want %d (alias preserves Dense kind tag)", h.Id[0], neuron.DENSE)
	}
	h.SetMiss(1.5)
	if got := *h.GetMiss(); got != 1.5 {
		t.Errorf("Hidden inherits Dense methods; got miss = %v", got)
	}
}

// TestNeuronTypeConstants documents the kind tags exposed by the neuron
// package — the test pins them so that a future renumber surfaces here
// rather than silently breaking JSON-encoded snapshots.
func TestNeuronTypeConstants(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		got  uint8
		want uint8
	}{
		{"UNKNOWN", neuron.UNKNOWN, 0},
		{"INPUT", neuron.INPUT, 1},
		{"OUTPUT", neuron.OUTPUT, 2},
		{"DENSE", neuron.DENSE, 3},
		{"BIAS", neuron.BIAS, 4},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("neuron.%s = %d; want %d", c.name, c.got, c.want)
		}
	}
}
