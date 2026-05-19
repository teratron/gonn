//go:build cgo && opencl

package opencl

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/teratron/gonn/pkg/compute"
	computecpu "github.com/teratron/gonn/pkg/compute/cpu"
	"github.com/teratron/gonn/pkg/utils"
)

// skipBenchIfNoDevice skips the benchmark when no OpenCL GPU is found.
func skipBenchIfNoDevice(b *testing.B) *openclBackend[float64] {
	b.Helper()
	backend, err := initialise[float64]()
	if errors.Is(err, utils.ErrBackendUnavailable) {
		b.Skip("no OpenCL device found; skipping")
	}
	if err != nil {
		b.Fatalf("initialise: %v", err)
	}
	return backend
}

// makeDenseFixture returns a LayerHandle and input slice for an inSize→outSize layer.
func makeDenseFixture(inSize, outSize int, seed uint64) (compute.LayerHandle[float64], []float64) {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	weights := make([][]float64, outSize)
	flat := make([]float64, outSize*inSize)
	for i := range flat {
		flat[i] = rng.NormFloat64() * 0.3
	}
	for i := range weights {
		weights[i] = flat[i*inSize : (i+1)*inSize]
	}
	bias := make([]float64, outSize)
	for i := range bias {
		bias[i] = rng.NormFloat64() * 0.1
	}
	input := make([]float64, inSize)
	for i := range input {
		input[i] = rng.NormFloat64()
	}
	sigmoid := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	dsigmoid := func(x float64) float64 { s := sigmoid(x); return s * (1 - s) }
	layer := compute.LayerHandle[float64]{
		Activation: sigmoid,
		Derivative: dsigmoid,
		Weights:    weights,
		Bias:       bias,
		Size:       outSize,
	}
	return layer, input
}

// BenchmarkDenseForward64 benchmarks OpenCL Dense Forward for a 64→64 layer.
func BenchmarkDenseForward64(b *testing.B) {
	benchmarkDenseForwardGPU(b, 64, 64, 1)
}

// BenchmarkDenseForward256 benchmarks OpenCL Dense Forward for a 256→256 layer.
//
// Expected speedup floor vs CPU reference (BenchmarkCPUDenseForward256):
//   - ≥2× for ≥256×256 matrices on Intel/AMD integrated GPU
//   - ≥5× on discrete NVIDIA/AMD GPU
//
// Actual throughput is hardware-dependent; the gate only requires this bench
// compiles and runs (T-16B03). Document baseline ns/op from CI notes in
// commit description when merging to main.
func BenchmarkDenseForward256(b *testing.B) {
	benchmarkDenseForwardGPU(b, 256, 256, 2)
}

// BenchmarkDenseForward1024 benchmarks OpenCL Dense Forward for a 1024→1024 layer.
func BenchmarkDenseForward1024(b *testing.B) {
	benchmarkDenseForwardGPU(b, 1024, 1024, 3)
}

func benchmarkDenseForwardGPU(b *testing.B, inSize, outSize int, seed uint64) {
	b.Helper()
	backend := skipBenchIfNoDevice(b)
	layer, input := makeDenseFixture(inSize, outSize, seed)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Forward(layer, input); err != nil {
			b.Fatalf("Forward: %v", err)
		}
	}
}

// BenchmarkDenseBackward64 benchmarks OpenCL Dense Backward for a 64→64 layer.
func BenchmarkDenseBackward64(b *testing.B) {
	benchmarkDenseBackwardGPU(b, 64, 64, 10)
}

// BenchmarkDenseBackward256 benchmarks OpenCL Dense Backward for a 256→256 layer.
func BenchmarkDenseBackward256(b *testing.B) {
	benchmarkDenseBackwardGPU(b, 256, 256, 11)
}

// BenchmarkDenseBackward1024 benchmarks OpenCL Dense Backward for a 1024→1024 layer.
func BenchmarkDenseBackward1024(b *testing.B) {
	benchmarkDenseBackwardGPU(b, 1024, 1024, 12)
}

func benchmarkDenseBackwardGPU(b *testing.B, inSize, outSize int, seed uint64) {
	b.Helper()
	backend := skipBenchIfNoDevice(b)
	layer, _ := makeDenseFixture(inSize, outSize, seed)
	gradient := make([]float64, outSize)
	rng := rand.New(rand.NewPCG(seed+100, seed+101))
	for i := range gradient {
		gradient[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Backward(layer, gradient); err != nil {
			b.Fatalf("Backward: %v", err)
		}
	}
}

// BenchmarkCPUDenseForward64 provides the CPU baseline for Forward 64→64.
func BenchmarkCPUDenseForward64(b *testing.B) {
	benchmarkDenseForwardCPU(b, 64, 64, 1)
}

// BenchmarkCPUDenseForward256 provides the CPU baseline for Forward 256→256.
func BenchmarkCPUDenseForward256(b *testing.B) {
	benchmarkDenseForwardCPU(b, 256, 256, 2)
}

// BenchmarkCPUDenseForward1024 provides the CPU baseline for Forward 1024→1024.
func BenchmarkCPUDenseForward1024(b *testing.B) {
	benchmarkDenseForwardCPU(b, 1024, 1024, 3)
}

func benchmarkDenseForwardCPU(b *testing.B, inSize, outSize int, seed uint64) {
	b.Helper()
	cpu := computecpu.Backend[float64]{}
	layer, input := makeDenseFixture(inSize, outSize, seed)
	b.ResetTimer()
	for range b.N {
		if _, err := cpu.Forward(layer, input); err != nil {
			b.Fatalf("CPU Forward: %v", err)
		}
	}
}

// BenchmarkCPUDenseBackward256 provides the CPU baseline for Backward 256→256.
func BenchmarkCPUDenseBackward256(b *testing.B) {
	cpu := computecpu.Backend[float64]{}
	layer, _ := makeDenseFixture(256, 256, 11)
	gradient := make([]float64, 256)
	rng := rand.New(rand.NewPCG(111, 112))
	for i := range gradient {
		gradient[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for range b.N {
		if _, err := cpu.Backward(layer, gradient); err != nil {
			b.Fatalf("CPU Backward: %v", err)
		}
	}
}
