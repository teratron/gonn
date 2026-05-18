package recurrent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

// TestSimpleRNNAPI covers InputSize, OutputSize, GradSlots, GradWhh, Step, ResetState.
func TestSimpleRNNAPI(t *testing.T) {
	t.Parallel()
	rng, _ := utils.NewRNG(300)
	r := NewSimpleRNN[float64](4, 3, 5)
	r.Init(rng)

	if got := r.InputSize(); got != 4*3 {
		t.Errorf("InputSize = %d, want %d", got, 4*3)
	}
	if got := r.OutputSize(); got != 4*5 {
		t.Errorf("OutputSize = %d, want %d", got, 4*5)
	}

	gW, gB := r.GradSlots()
	if len(gW) != 5*3 {
		t.Errorf("GradSlots gradW len = %d, want %d", len(gW), 5*3)
	}
	if len(gB) != 5 {
		t.Errorf("GradSlots gradB len = %d, want %d", len(gB), 5)
	}
	if len(r.GradWhh()) != 5*5 {
		t.Errorf("GradWhh len = %d, want %d", len(r.GradWhh()), 5*5)
	}

	// Step API: run two independent steps and check output length.
	x := make([]float64, 3)
	h1 := r.Step(x)
	if len(h1) != 5 {
		t.Errorf("Step output len = %d, want %d", len(h1), 5)
	}
	r.ResetState()
	h2 := r.Step(x)
	// After reset the state is zero, so h2 should equal h1 (same x and zero hidden).
	for i := range h1 {
		if h1[i] != h2[i] {
			t.Errorf("Step after ResetState differs at index %d: %v vs %v", i, h2[i], h1[i])
		}
	}
}

// TestSimpleRNNStepPanic verifies Step panics on a zero-value layer (Wxh == nil).
func TestSimpleRNNStepPanic(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Error("Step on zero-value layer should panic")
		}
	}()
	var r SimpleRNN[float64]
	r.Hidden = 2
	r.Step(make([]float64, 2))
}

// TestSimpleRNNUnmarshalErrors checks that UnmarshalJSON rejects malformed payloads.
func TestSimpleRNNUnmarshalErrors(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		payload string
	}{
		{"corrupted JSON", `{bad json`},
		{"wrong Wxh length", `{"type":"SimpleRNN","seq_len":2,"in_size":3,"hidden":4,"wxh":[0],"whh":[],"bh":[]}`},
		{"wrong Whh length", `{"type":"SimpleRNN","seq_len":2,"in_size":3,"hidden":4,"wxh":` + zeros(12) + `,"whh":[0],"bh":[]}`},
		{"wrong Bh length", `{"type":"SimpleRNN","seq_len":2,"in_size":3,"hidden":4,"wxh":` + zeros(12) + `,"whh":` + zeros(16) + `,"bh":[0,0]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var r SimpleRNN[float64]
			if err := json.Unmarshal([]byte(tc.payload), &r); err == nil {
				t.Errorf("UnmarshalJSON(%q) expected error, got nil", tc.name)
			}
		})
	}
}

// TestLSTMAPI covers InputSize, OutputSize, GradSlots, GradWh, Step, ResetState.
func TestLSTMAPI(t *testing.T) {
	t.Parallel()
	rng, _ := utils.NewRNG(400)
	l := NewLSTM[float64](4, 3, 5)
	l.Init(rng)

	if got := l.InputSize(); got != 4*3 {
		t.Errorf("InputSize = %d, want %d", got, 4*3)
	}
	if got := l.OutputSize(); got != 4*5 {
		t.Errorf("OutputSize = %d, want %d", got, 4*5)
	}

	gW, gB := l.GradSlots()
	if len(gW) != 4*5*3 {
		t.Errorf("GradSlots gradW len = %d, want %d", len(gW), 4*5*3)
	}
	if len(gB) != 4*5 {
		t.Errorf("GradSlots gradB len = %d, want %d", len(gB), 4*5)
	}
	if len(l.GradWh()) != 4*5*5 {
		t.Errorf("GradWh len = %d, want %d", len(l.GradWh()), 4*5*5)
	}

	x := make([]float64, 3)
	h1 := l.Step(x)
	if len(h1) != 5 {
		t.Errorf("Step output len = %d, want %d", len(h1), 5)
	}
	l.ResetState()
	h2 := l.Step(x)
	for i := range h1 {
		if h1[i] != h2[i] {
			t.Errorf("Step after ResetState differs at index %d: %v vs %v", i, h2[i], h1[i])
		}
	}
}

// TestLSTMStepPanic verifies Step panics on a zero-value layer (Wx == nil).
func TestLSTMStepPanic(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Error("Step on zero-value LSTM should panic")
		}
	}()
	var l LSTM[float64]
	l.Hidden = 2
	l.Step(make([]float64, 2))
}

// TestLSTMUnmarshalErrors checks that UnmarshalJSON rejects malformed payloads.
func TestLSTMUnmarshalErrors(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		payload string
	}{
		{"corrupted JSON", `{bad json`},
		{"wrong Wx length", `{"type":"LSTM","seq_len":2,"in_size":3,"hidden":4,"wx":[0],"wh":[],"b":[]}`},
		{"wrong Wh length", `{"type":"LSTM","seq_len":2,"in_size":3,"hidden":4,"wx":` + zeros(48) + `,"wh":[0],"b":[]}`},
		{"wrong B length", `{"type":"LSTM","seq_len":2,"in_size":3,"hidden":4,"wx":` + zeros(48) + `,"wh":` + zeros(64) + `,"b":[0]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var l LSTM[float64]
			if err := json.Unmarshal([]byte(tc.payload), &l); err == nil {
				t.Errorf("UnmarshalJSON(%q) expected error, got nil", tc.name)
			}
		})
	}
}

// TestItoaHelper exercises the itoa helper via the error message path.
func TestItoaHelper(t *testing.T) {
	t.Parallel()
	cases := []struct {
		n    int
		want string
	}{
		{0, "0"},
		{1, "1"},
		{-1, "-1"},
		{123, "123"},
		{-456, "-456"},
		{1000000, "1000000"},
	}
	for _, tc := range cases {
		if got := itoa(tc.n); got != tc.want {
			t.Errorf("itoa(%d) = %q, want %q", tc.n, got, tc.want)
		}
	}
}

// zeros returns a JSON array of n zero floats, e.g. "[0,0,0]".
func zeros(n int) string {
	if n == 0 {
		return "[]"
	}
	var b strings.Builder
	b.WriteByte('[')
	for i := range n {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteByte('0')
	}
	b.WriteByte(']')
	return b.String()
}
