package main

import (
	"sync/atomic"
	"testing"

	"github.com/teratron/gonn/pkg/nn"
)

// TestCallbacksFireDuringFit asserts that both EpochCallback and
// BatchCallback are invoked at least once during a short Fit. This
// catches regressions where a callback wiring change silently drops the
// hook on the floor — a class of bug the XOR convergence test would not
// surface because Fit can succeed without ever invoking the callbacks.
func TestCallbacksFireDuringFit(t *testing.T) {
	var (
		epochCalls atomic.Int64
		batchCalls atomic.Int64
	)
	n, err := nn.New[float32](
		nn.PresetXOR[float32](),
		nn.WithMaxIterations[float32](200),
		nn.WithLossLimit[float32](-1), // run the full loop
		nn.WithEpochCallback[float32](func(uint, float32) { epochCalls.Add(1) }),
		nn.WithBatchCallback[float32](func(uint, float32) { batchCalls.Add(1) }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := n.Fit(xorDataset()); err != nil {
		t.Fatal(err)
	}
	if epochCalls.Load() == 0 {
		t.Error("EpochCallback never fired")
	}
	if batchCalls.Load() == 0 {
		t.Error("BatchCallback never fired")
	}
}

// TestTrainConverges keeps the smoke-test parity with E01 — the network
// should still cross the loose convergence threshold even with the
// callback overhead.
func TestTrainConverges(t *testing.T) {
	if l := train(); l > 0.15 {
		t.Errorf("final loss = %v, want < 0.15", l)
	}
}
