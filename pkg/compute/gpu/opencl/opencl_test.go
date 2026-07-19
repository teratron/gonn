//go:build cgo && opencl

package opencl

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// skipIfNoDevice skips the test when no OpenCL GPU is found.
func skipIfNoDevice(t *testing.T) *openclBackend[float64] {
	t.Helper()
	b, err := initialise[float64]()
	if errors.Is(err, utils.ErrBackendUnavailable) {
		t.Skip("no OpenCL device found; skipping")
	}
	if err != nil {
		t.Fatalf("initialise: %v", err)
	}
	return b
}

// TestBufferRoundTrip verifies Allocate → Write → Read recovers the source.
func TestBufferRoundTrip(t *testing.T) {
	b := skipIfNoDevice(t)

	const size = 64
	rng := rand.New(rand.NewPCG(1, 2))
	src := make([]float64, size)
	for i := range src {
		src[i] = rng.NormFloat64()
	}

	buf, err := b.Allocate(size)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	defer b.Free(buf)

	if err := b.Write(buf, src); err != nil {
		t.Fatalf("Write: %v", err)
	}

	dst := make([]float64, size)
	if err := b.Read(buf, dst); err != nil {
		t.Fatalf("Read: %v", err)
	}

	for i := range src {
		if src[i] != dst[i] {
			t.Errorf("round-trip mismatch at [%d]: got %v, want %v", i, dst[i], src[i])
		}
	}
}

// TestDenseForwardVsCPU cross-references the OpenCL Dense Forward kernel
// against the reference CPU path within the numerical tolerance contract.
func TestDenseForwardVsCPU(t *testing.T) {
	b := skipIfNoDevice(t)

	cases := []struct{ inSize, outSize int }{
		{8, 4},
		{64, 10},
	}

	for _, tc := range cases {
		rng := rand.New(rand.NewPCG(99, uint64(tc.inSize)))
		weights := make([][]float64, tc.outSize)
		flat := make([]float64, tc.outSize*tc.inSize)
		for i := range flat {
			flat[i] = rng.NormFloat64() * 0.3
		}
		for i := range weights {
			weights[i] = flat[i*tc.inSize : (i+1)*tc.inSize]
		}
		bias := make([]float64, tc.outSize)
		for i := range bias {
			bias[i] = rng.NormFloat64() * 0.1
		}
		input := make([]float64, tc.inSize)
		for i := range input {
			input[i] = rng.NormFloat64()
		}

		sigmoid := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
		layer := compute.LayerHandle[float64]{
			Activation: sigmoid,
			Weights:    weights,
			Bias:       bias,
			Size:       tc.outSize,
		}

		// Reference CPU forward pass.
		cpuOut := make([]float64, tc.outSize)
		for i, row := range weights {
			var acc float64
			for j, w := range row {
				acc += w * input[j]
			}
			cpuOut[i] = sigmoid(acc + bias[i])
		}

		// OpenCL forward pass.
		gpuOut, err := b.Forward(layer, input)
		if err != nil {
			t.Fatalf("Forward(%d→%d): %v", tc.inSize, tc.outSize, err)
		}
		if len(gpuOut) != tc.outSize {
			t.Fatalf("Forward output len = %d, want %d", len(gpuOut), tc.outSize)
		}

		const tolF64 = 1e-12
		for i := range cpuOut {
			if diff := math.Abs(gpuOut[i] - cpuOut[i]); diff > tolF64 {
				t.Errorf("inSize=%d outSize=%d output[%d]: gpu=%.15g cpu=%.15g diff=%.3e",
					tc.inSize, tc.outSize, i, gpuOut[i], cpuOut[i], diff)
			}
		}
	}
}
