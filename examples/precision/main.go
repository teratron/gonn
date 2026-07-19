// float32 vs float64 parity.
//
// Builds the XOR network twice (once at each numeric precision) and
// reports the final loss + elapsed time per run. Both runs share the
// same topology and hyperparameters; the only difference is the type
// parameter T flowing through the generics, so any drift between them
// comes from float precision alone.
package main

import (
	"fmt"
	"time"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	loss32, dt32 := trainF32()
	loss64, dt64 := trainF64()

	fmt.Println()
	fmt.Println("Precision parity")
	fmt.Printf("  float32  loss=%.6f  elapsed=%s\n", loss32, dt32)
	fmt.Printf("  float64  loss=%.6f  elapsed=%s\n", loss64, dt64)
}

// trainF32 builds and fits the canonical XOR network at single precision.
// Returns final loss and wall-clock duration so the test (and stdout)
// can compare runs side by side without re-implementing the timing.
func trainF32() (float32, time.Duration) {
	start := time.Now()
	n, err := nn.NewBuilder[float32]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithLearningRate(0.3).
		WithLoss(loss.MSE).
		WithMaxIterations(10_000).
		WithLossLimit(1e-4).
		Compile()
	if err != nil {
		fmt.Printf("f32 Compile failed: %v\n", err)
		return 1, time.Since(start)
	}
	dataset := []nn.Sample[float32]{
		{Input: []float32{0, 0}, Target: []float32{0}},
		{Input: []float32{0, 1}, Target: []float32{1}},
		{Input: []float32{1, 0}, Target: []float32{1}},
		{Input: []float32{1, 1}, Target: []float32{0}},
	}
	_, l, _ := n.Fit(dataset)
	return l, time.Since(start)
}

// trainF64 mirrors trainF32 at double precision. The duplication is
// intentional — generic T cannot be picked at runtime, so each precision
// needs its own concrete entry point.
func trainF64() (float64, time.Duration) {
	start := time.Now()
	n, err := nn.NewBuilder[float64]().
		Input(2).
		Dense(4, activation.SIGMOID, true).
		Output(1, activation.SIGMOID, true).
		WithLearningRate(0.3).
		WithLoss(loss.MSE).
		WithMaxIterations(10_000).
		WithLossLimit(1e-4).
		Compile()
	if err != nil {
		fmt.Printf("f64 Compile failed: %v\n", err)
		return 1, time.Since(start)
	}
	dataset := []nn.Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
	_, l, _ := n.Fit(dataset)
	return l, time.Since(start)
}
