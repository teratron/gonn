package nn

import (
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// stubUnavailableBackend is a test-local backend that always returns
// ErrBackendUnavailable, simulating a GPU device that is not present.
type stubUnavailableBackend[T utils.Float] struct{}

func (s *stubUnavailableBackend[T]) Name() string { return "stub-unavailable" }
func (s *stubUnavailableBackend[T]) Forward(_ compute.LayerHandle[T], _ []T) ([]T, error) {
	return nil, utils.ErrBackendUnavailable
}
func (s *stubUnavailableBackend[T]) Backward(_ compute.LayerHandle[T], _ []T) ([]T, error) {
	return nil, utils.ErrBackendUnavailable
}
func (s *stubUnavailableBackend[T]) UpdateWeights(_ compute.LayerHandle[T], _, _ []T, _ T) error {
	return utils.ErrBackendUnavailable
}
func (s *stubUnavailableBackend[T]) Allocate(_ int) (compute.Buffer[T], error) {
	return compute.Buffer[T]{}, utils.ErrBackendUnavailable
}
func (s *stubUnavailableBackend[T]) Free(_ compute.Buffer[T]) error {
	return utils.ErrBackendUnavailable
}

// Compile-time: stubUnavailableBackend must satisfy compute.Backend.
var (
	_ compute.Backend[float32] = (*stubUnavailableBackend[float32])(nil)
	_ compute.Backend[float64] = (*stubUnavailableBackend[float64])(nil)
)

// TestWithBackend verifies the compile-time backend probe and CPU fallback
// (l1-compute-backend §5.3 graceful fallback, T-16B02).
func TestWithBackend(t *testing.T) {
	t.Parallel()

	baseOpts := func() []Option[float64] {
		return []Option[float64]{
			WithInput[float64](2),
			WithHiddenLayer[float64](4, activation.SIGMOID),
			WithOutput[float64](1, activation.SIGMOID),
			// Explicit seed so the parity subtest's two networks get identical
			// weights deterministically. Without it both use the default seed 0
			// (wall-clock) and parity only held when both compiles landed in the
			// same clock tick.
			WithWeightInitSeed[float64](42),
		}
	}

	t.Run("nil_backend_defaults_to_cpu", func(t *testing.T) {
		t.Parallel()
		n, err := New(baseOpts()...)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if n.backend == nil {
			t.Fatal("backend must not be nil after compile (CPU default)")
		}
		if got := n.backend.Name(); got != "cpu" {
			t.Errorf("backend name = %q, want cpu", got)
		}
	})

	t.Run("unavailable_backend_falls_back_to_cpu", func(t *testing.T) {
		t.Parallel()
		opts := append(baseOpts(), WithBackend(&stubUnavailableBackend[float64]{}))
		n, err := New(opts...)
		if err != nil {
			t.Fatalf("New with unavailable backend: %v", err)
		}
		// After fallback the backend must be the CPU reference.
		if n.backend == nil {
			t.Fatal("backend must not be nil after fallback")
		}
		if got := n.backend.Name(); got != "cpu" {
			t.Errorf("backend name = %q, want cpu after ErrBackendUnavailable fallback", got)
		}
	})

	t.Run("result_parity_cpu_vs_fallback", func(t *testing.T) {
		t.Parallel()
		// Both networks share weight-init seed 0 (compile hard-codes it), so
		// weights are identical — Query results must match.
		n1, err := New(baseOpts()...)
		if err != nil {
			t.Fatalf("New (CPU): %v", err)
		}
		opts2 := append(baseOpts(), WithBackend(&stubUnavailableBackend[float64]{}))
		n2, err := New(opts2...)
		if err != nil {
			t.Fatalf("New (fallback): %v", err)
		}

		input := []float64{0.5, 0.5}
		out1, err := n1.Query(input)
		if err != nil {
			t.Fatalf("Query n1: %v", err)
		}
		out2, err := n2.Query(input)
		if err != nil {
			t.Fatalf("Query n2 (fallback): %v", err)
		}
		if len(out1) != len(out2) {
			t.Fatalf("output length mismatch: %d vs %d", len(out1), len(out2))
		}
		for i := range out1 {
			if out1[i] != out2[i] {
				t.Errorf("output[%d] = %v (cpu) vs %v (fallback) — mismatch", i, out1[i], out2[i])
			}
		}
	})
}
