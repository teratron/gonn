package layer

import (
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron"
)

func TestNewInputCarriesSize(t *testing.T) {
	t.Parallel()
	in := NewInput[float64](4)
	if in.Size != 4 {
		t.Errorf("Size = %d; want 4 (regression of legacy NewInput passing 0 to Init)", in.Size)
	}
	if got := len(in.Cells()); got != 4 {
		t.Errorf("len(Cells) = %d; want 4", got)
	}
	if in.Type != neuron.INPUT {
		t.Errorf("Type = %d; want %d", in.Type, neuron.INPUT)
	}
	for idx, c := range in.Cells() {
		if c == nil {
			t.Fatalf("input cell %d not allocated", idx)
		}
	}
}

func TestNewInputClampsNegativeSize(t *testing.T) {
	t.Parallel()
	in := NewInput[float64](-3)
	if in.Size != 0 {
		t.Errorf("negative size must clamp to 0, got %d", in.Size)
	}
	if len(in.Cells()) != 0 {
		t.Errorf("Cells() must be empty for size=0")
	}
}

func TestInputReinitChangesSize(t *testing.T) {
	t.Parallel()
	in := NewInput[float64](2)
	in.Init(5)
	if in.Size != 5 || len(in.Cells()) != 5 {
		t.Errorf("after Init(5): Size=%d, len(Cells)=%d; want 5/5", in.Size, len(in.Cells()))
	}
}

func TestNewDenseDoesNotPanicOnInit(t *testing.T) {
	t.Parallel()
	// Legacy code allocated &Dense{} with nil *base, which made any field
	// access on Init panic. The rewritten constructor must produce a
	// fully wired struct out of the box.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("NewDense panicked (regression of nil-deref defect): %v", r)
		}
	}()
	d := NewDense[float64](3, activation.SIGMOID, true)
	if d.Size != 3 {
		t.Errorf("Size = %d; want 3", d.Size)
	}
	if !d.Bias {
		t.Errorf("Bias = false; want true")
	}
	if d.Activation != activation.SIGMOID {
		t.Errorf("Activation = %v; want SIGMOID", d.Activation)
	}
	if d.BiasCell() == nil {
		t.Errorf("BiasCell must be allocated when Bias=true")
	}
	if got := *d.BiasCell().GetValue(); got != 1.0 {
		t.Errorf("Bias cell value = %v; want 1.0", got)
	}
}

func TestNewDenseWithoutBiasOmitsCell(t *testing.T) {
	t.Parallel()
	d := NewDense[float64](2, activation.ReLU, false)
	if d.BiasCell() != nil {
		t.Errorf("BiasCell must be nil when Bias=false")
	}
}

func TestDenseInitDeduplicatesBaseLogic(t *testing.T) {
	t.Parallel()
	d := NewDense[float64](2, activation.ReLU, false)
	d.Init(7, activation.TanH, true)
	// Reinit must propagate every field — verifies that Dense.Init
	// delegates rather than re-implementing.
	if d.Size != 7 {
		t.Errorf("Size after Init = %d; want 7", d.Size)
	}
	if d.Activation != activation.TanH {
		t.Errorf("Activation after Init = %v; want TANH", d.Activation)
	}
	if !d.Bias || d.BiasCell() == nil {
		t.Errorf("Bias state after Init: Bias=%v, BiasCell=%v", d.Bias, d.BiasCell())
	}
	if got := len(d.Cells()); got != 7 {
		t.Errorf("len(Cells) after Init = %d; want 7", got)
	}
}

func TestNewOutputDoesNotPanicOnInit(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("NewOutput panicked (regression of nil-deref defect): %v", r)
		}
	}()
	o := NewOutput[float64](2, activation.SOFTMAX, loss.MSE, true)
	if o.Size != 2 {
		t.Errorf("Size = %d; want 2", o.Size)
	}
	if o.Loss != loss.MSE {
		t.Errorf("Loss = %v; want MSE", o.Loss)
	}
	if o.Type != neuron.OUTPUT {
		t.Errorf("Type = %d; want %d", o.Type, neuron.OUTPUT)
	}
	if got := len(o.Cells()); got != 2 {
		t.Errorf("len(Cells) = %d; want 2", got)
	}
	if got := len(o.Targets()); got != 2 {
		t.Errorf("len(Targets) = %d; want 2", got)
	}
	for idx, c := range o.Cells() {
		if c == nil {
			t.Fatalf("output cell %d not allocated", idx)
		}
		if c.GetTarget() != &o.Targets()[idx] {
			t.Errorf("output cell %d target pointer mismatch", idx)
		}
	}
}

func TestOutputSetTarget(t *testing.T) {
	t.Parallel()
	o := NewOutput[float64](3, activation.SIGMOID, loss.MSE, false)
	o.SetTarget(1, 0.75)
	if got := o.Targets()[1]; got != 0.75 {
		t.Errorf("after SetTarget(1, 0.75), Targets[1] = %v; want 0.75", got)
	}
	// Pointer aliasing — the cell sees the new value through its target ptr.
	if got := *o.Cells()[1].GetTarget(); got != 0.75 {
		t.Errorf("output cell did not see SetTarget update: %v", got)
	}
}

func TestOutputInitClampsNegative(t *testing.T) {
	t.Parallel()
	o := NewOutput[float64](-1, activation.SIGMOID, loss.MSE, false)
	if o.Size != 0 || len(o.Cells()) != 0 || len(o.Targets()) != 0 {
		t.Errorf("negative size must clamp; got Size=%d, cells=%d, targets=%d",
			o.Size, len(o.Cells()), len(o.Targets()))
	}
	o.Init(-5, activation.SIGMOID, loss.MSE, false)
	if o.Size != 0 {
		t.Errorf("Init(-5) must also clamp to 0, got %d", o.Size)
	}
}

func TestOutputForwardWithLayerCells(t *testing.T) {
	t.Parallel()
	// Smoke check: cells produced by Output.populate must run forward
	// without recursion (regression of C-001 cell-side fix).
	o := NewOutput[float64](1, activation.SIGMOID, loss.MSE, false)
	o.SetTarget(0, 1.0)
	c := o.Cells()[0]
	c.CalculateValue() // No axons — value stays 0; miss = target - value = 1.
	if got := *c.GetMiss(); got != 1.0 {
		t.Errorf("Output cell miss after forward = %v; want 1.0", got)
	}
}

func TestSetCellOnDense(t *testing.T) {
	t.Parallel()
	d := NewDense[float64](2, activation.ReLU, false)
	original := d.Cells()[0]
	d.SetCell(0, original)
	if d.Cells()[0] != original {
		t.Errorf("SetCell did not write the supplied value")
	}
}
