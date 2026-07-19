package recurrent

import (
	"encoding/json"
	"testing"
)

func TestLastStepForward(t *testing.T) {
	t.Parallel()
	ls := NewLastStep[float64](4, 3)

	// Input: 4 timesteps × 3 features
	input := []float64{
		1, 2, 3, // t=0
		4, 5, 6, // t=1
		7, 8, 9, // t=2
		10, 11, 12, // t=3 (last)
	}
	out := ls.Forward(input)
	if len(out) != 3 {
		t.Fatalf("Forward length = %d, want 3", len(out))
	}
	want := []float64{10, 11, 12}
	for i, v := range want {
		if out[i] != v {
			t.Errorf("Forward[%d] = %v, want %v", i, out[i], v)
		}
	}
}

func TestLastStepBackward(t *testing.T) {
	t.Parallel()
	ls := NewLastStep[float64](4, 3)

	upstream := []float64{1, 2, 3}
	grad := ls.Backward(upstream)

	if len(grad) != 4*3 {
		t.Fatalf("Backward length = %d, want %d", len(grad), 4*3)
	}
	// First 3*3=9 elements must be zero; last 3 must equal upstream.
	for i := range 9 {
		if grad[i] != 0 {
			t.Errorf("grad[%d] = %v, want 0 (non-final timestep)", i, grad[i])
		}
	}
	for i, v := range upstream {
		if grad[9+i] != v {
			t.Errorf("grad[%d] = %v, want %v (final timestep)", 9+i, grad[9+i], v)
		}
	}
}

func TestLastStepRoundTrip(t *testing.T) {
	t.Parallel()
	ls := NewLastStep[float64](5, 8)
	data, err := json.Marshal(ls)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	var ls2 LastStep[float64]
	if err := json.Unmarshal(data, &ls2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if ls2.SeqLen != ls.SeqLen || ls2.Hidden != ls.Hidden {
		t.Errorf("shape mismatch after round-trip: got (%d,%d), want (%d,%d)",
			ls2.SeqLen, ls2.Hidden, ls.SeqLen, ls.Hidden)
	}
}

func TestLastStepSizes(t *testing.T) {
	t.Parallel()
	ls := NewLastStep[float32](7, 4)
	if ls.InputSize() != 7*4 {
		t.Errorf("InputSize = %d, want %d", ls.InputSize(), 7*4)
	}
	if ls.OutputSize() != 4 {
		t.Errorf("OutputSize = %d, want 4", ls.OutputSize())
	}
	gW, gB := ls.GradSlots()
	if gW != nil || gB != nil {
		t.Errorf("GradSlots must return (nil, nil) for stateless layer")
	}
}
