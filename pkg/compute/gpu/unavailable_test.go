package gpu

import (
	"errors"
	"testing"

	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// TestUnavailableBackendName verifies Name() returns the vendor string.
func TestUnavailableBackendName(t *testing.T) {
	t.Parallel()
	b := &unavailableBackend[float64]{vendor: "test-vendor"}
	if got := b.Name(); got != "test-vendor" {
		t.Errorf("Name() = %q, want %q", got, "test-vendor")
	}
}

// TestUnavailableBackendMethods checks that every method returns ErrBackendUnavailable.
func TestUnavailableBackendMethods(t *testing.T) {
	t.Parallel()
	b := &unavailableBackend[float64]{vendor: VendorOpenCL}
	handle := compute.LayerHandle[float64]{}

	t.Run("Forward", func(t *testing.T) {
		t.Parallel()
		out, err := b.Forward(handle, nil)
		if out != nil {
			t.Errorf("Forward: want nil output, got %v", out)
		}
		if !errors.Is(err, utils.ErrBackendUnavailable) {
			t.Errorf("Forward: want ErrBackendUnavailable, got %v", err)
		}
	})

	t.Run("Backward", func(t *testing.T) {
		t.Parallel()
		out, err := b.Backward(handle, nil)
		if out != nil {
			t.Errorf("Backward: want nil output, got %v", out)
		}
		if !errors.Is(err, utils.ErrBackendUnavailable) {
			t.Errorf("Backward: want ErrBackendUnavailable, got %v", err)
		}
	})

	t.Run("UpdateWeights", func(t *testing.T) {
		t.Parallel()
		err := b.UpdateWeights(handle, nil, nil, 0)
		if !errors.Is(err, utils.ErrBackendUnavailable) {
			t.Errorf("UpdateWeights: want ErrBackendUnavailable, got %v", err)
		}
	})

	t.Run("Allocate", func(t *testing.T) {
		t.Parallel()
		buf, err := b.Allocate(64)
		if buf.Data != nil {
			t.Errorf("Allocate: want nil buffer data, got %v", buf.Data)
		}
		if !errors.Is(err, utils.ErrBackendUnavailable) {
			t.Errorf("Allocate: want ErrBackendUnavailable, got %v", err)
		}
	})

	t.Run("Free", func(t *testing.T) {
		t.Parallel()
		err := b.Free(compute.Buffer[float64]{})
		if !errors.Is(err, utils.ErrBackendUnavailable) {
			t.Errorf("Free: want ErrBackendUnavailable, got %v", err)
		}
	})
}

// TestNewUnregisteredVendor verifies gpu.New returns ErrBackendUnavailable for
// any vendor not currently registered (no OpenCL/CUDA build tags in this test run).
func TestNewUnregisteredVendor(t *testing.T) {
	t.Parallel()
	for _, vendor := range []string{VendorOpenCL, VendorCUDA, "nonexistent"} {
		_, err := New[float64](vendor)
		if !errors.Is(err, utils.ErrBackendUnavailable) {
			t.Errorf("New(%q) error = %v, want ErrBackendUnavailable", vendor, err)
		}
	}
}

// TestUnavailableBackendFloat32 exercises the float32 specialisation.
func TestUnavailableBackendFloat32(t *testing.T) {
	t.Parallel()
	b := &unavailableBackend[float32]{vendor: VendorCUDA}
	if b.Name() != VendorCUDA {
		t.Errorf("Name() = %q, want %q", b.Name(), VendorCUDA)
	}
	_, err := New[float32](VendorOpenCL)
	if !errors.Is(err, utils.ErrBackendUnavailable) {
		t.Errorf("New[float32] error = %v, want ErrBackendUnavailable", err)
	}
}
