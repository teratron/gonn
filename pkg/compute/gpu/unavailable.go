package gpu

import (
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// New resolves vendor into a Backend[T] instance via the compute registry.
// When the vendor is not registered (build tags absent or no device present),
// New returns nil and [utils.ErrBackendUnavailable]. The caller should
// fall back to the CPU backend in that case.
//
// AI-Meta:
//   - Purpose: Resolve a GPU vendor string into a Backend[T]; returns ErrBackendUnavailable when the backend is absent.
//   - Usage: b, err := gpu.New[float32](gpu.VendorOpenCL); if errors.Is(err, utils.ErrBackendUnavailable) { ... }.
//   - Errors: ErrBackendUnavailable (vendor not registered or no device), ErrUserConfig (unknown vendor string).
//   - Concurrency: Safe; read-only registry lookup.
//   - Related: [VendorOpenCL], [VendorCUDA], [compute.Get].
//   - Stability: Stable.
func New[T utils.Float](vendor string) (compute.Backend[T], error) {
	b, err := compute.Get[T](vendor)
	if err != nil {
		return nil, utils.ErrBackendUnavailable
	}
	return b, nil
}

// unavailableBackend[T] is the zero-value stub satisfying [compute.Backend[T]].
// Sub-packages (opencl, cuda) return it when device discovery finds zero
// devices at runtime, as distinct from the build-tag-absent case handled by
// New above. All methods return [utils.ErrBackendUnavailable].
type unavailableBackend[T utils.Float] struct {
	vendor string
}

func (u *unavailableBackend[T]) Name() string { return u.vendor }

func (u *unavailableBackend[T]) Forward(_ compute.LayerHandle[T], _ []T) ([]T, error) {
	return nil, utils.ErrBackendUnavailable
}

func (u *unavailableBackend[T]) Backward(_ compute.LayerHandle[T], _ []T) ([]T, error) {
	return nil, utils.ErrBackendUnavailable
}

func (u *unavailableBackend[T]) UpdateWeights(_ compute.LayerHandle[T], _, _ []T, _ T) error {
	return utils.ErrBackendUnavailable
}

func (u *unavailableBackend[T]) Allocate(_ int) (compute.Buffer[T], error) {
	return compute.Buffer[T]{}, utils.ErrBackendUnavailable
}

func (u *unavailableBackend[T]) Free(_ compute.Buffer[T]) error {
	return utils.ErrBackendUnavailable
}

// Compile-time assertion: unavailableBackend must satisfy compute.Backend.
var (
	_ compute.Backend[float32] = (*unavailableBackend[float32])(nil)
	_ compute.Backend[float64] = (*unavailableBackend[float64])(nil)
)
