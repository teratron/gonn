//go:build cgo && opencl

// Package opencl provides an OpenCL 1.2+ compute backend for GoNN.
//
// Import this package with a blank import to register the "opencl" backend:
//
//	import _ "github.com/teratron/gonn/pkg/compute/gpu/opencl"
//
// Build with -tags opencl (and cgo enabled, which is the default).
//
// AI-Meta:
//   - Purpose: OpenCL 1.2 Backend[T] implementation; registers under gpu.VendorOpenCL at init.
//   - Usage: import _ "github.com/teratron/gonn/pkg/compute/gpu/opencl"; gpu.New[float32](gpu.VendorOpenCL).
//   - Errors: ErrBackendUnavailable (no device), ErrBackendKernel (kernel launch failed), ErrBackendTransfer.
//   - Concurrency: NotSafe; each Backend instance owns its queue.
//   - Related: [compute.Backend], [gpu.New], [gpu.VendorOpenCL].
//   - Stability: Experimental.
package opencl

// #include <CL/cl.h>
import "C"

import (
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

func init() {
	compute.Register[float32](vendorName, newFactory[float32])
	compute.Register[float64](vendorName, newFactory[float64])
}

const vendorName = "opencl"

// openclBackend[T] is the OpenCL Backend[T] implementation.
type openclBackend[T utils.Float] struct {
	ctx     C.cl_context
	queue   C.cl_command_queue
	program C.cl_program
	reg     *bufferRegistry
}

// Compile-time assertion: openclBackend must satisfy compute.Backend.
var (
	_ compute.Backend[float32] = (*openclBackend[float32])(nil)
	_ compute.Backend[float64] = (*openclBackend[float64])(nil)
)

// newFactory attempts OpenCL platform/device discovery and returns an initialised
// backend, or returns nil when no device is present (causing gpu.New to return
// ErrBackendUnavailable via the registry miss path).
func newFactory[T utils.Float]() compute.Backend[T] {
	b, err := initialise[T]()
	if err != nil {
		return nil
	}
	return b
}

// initialise runs the OpenCL platform and device handshake.
// Returns ErrBackendUnavailable when no platform or device is found.
func initialise[T utils.Float]() (*openclBackend[T], error) {
	// Query platform count.
	var numPlatforms C.cl_uint
	if ret := C.clGetPlatformIDs(0, nil, &numPlatforms); ret != C.CL_SUCCESS || numPlatforms == 0 {
		return nil, utils.ErrBackendUnavailable
	}

	platforms := make([]C.cl_platform_id, numPlatforms)
	if ret := C.clGetPlatformIDs(numPlatforms, &platforms[0], nil); ret != C.CL_SUCCESS {
		return nil, utils.ErrBackendUnavailable
	}

	// Pick first platform with a GPU device.
	var device C.cl_device_id
	found := false
	for _, plat := range platforms {
		var numDevices C.cl_uint
		if C.clGetDeviceIDs(plat, C.CL_DEVICE_TYPE_GPU, 0, nil, &numDevices) != C.CL_SUCCESS || numDevices == 0 {
			continue
		}
		devices := make([]C.cl_device_id, 1)
		if C.clGetDeviceIDs(plat, C.CL_DEVICE_TYPE_GPU, 1, &devices[0], nil) != C.CL_SUCCESS {
			continue
		}
		device = devices[0]
		found = true
		break
	}
	if !found {
		return nil, utils.ErrBackendUnavailable
	}

	// Create context and command queue.
	var errCode C.cl_int
	ctx := C.clCreateContext(nil, 1, &device, nil, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, utils.ErrBackendUnavailable
	}
	queue := C.clCreateCommandQueue(ctx, device, 0, &errCode)
	if errCode != C.CL_SUCCESS {
		C.clReleaseContext(ctx)
		return nil, utils.ErrBackendUnavailable
	}

	return &openclBackend[T]{
		ctx:   ctx,
		queue: queue,
		reg:   newRegistry(),
	}, nil
}

// Name identifies the backend in registry lookups and log messages.
func (b *openclBackend[T]) Name() string { return vendorName }

// Allocate creates a device-side buffer of size T elements.
func (b *openclBackend[T]) Allocate(size int) (compute.Buffer[T], error) {
	return allocBuf[T](b.reg, b.ctx, size)
}

// Free releases the device-side buffer.
func (b *openclBackend[T]) Free(buf compute.Buffer[T]) error {
	return freeBuf[T](b.reg, buf)
}

// Write transfers src to the device buffer. Exposed for TestBufferRoundTrip.
func (b *openclBackend[T]) Write(buf compute.Buffer[T], src []T) error {
	return writeBuf[T](b.reg, b.queue, buf, src)
}

// Read transfers device buffer contents to dst. Exposed for TestBufferRoundTrip.
func (b *openclBackend[T]) Read(buf compute.Buffer[T], dst []T) error {
	return readBuf[T](b.reg, b.queue, buf, dst)
}

// Forward runs the Dense Forward kernel (implemented in T-15B03).
// Returns ErrBackendUnavailable until Phase C kernel is compiled in.
func (b *openclBackend[T]) Forward(layer compute.LayerHandle[T], input []T) ([]T, error) {
	if b.program == nil {
		return nil, utils.ErrBackendUnavailable
	}
	return kernelDenseForward(b, layer, input)
}

// Backward is not implemented in Phase 15; deferred to Phase 16.
func (b *openclBackend[T]) Backward(_ compute.LayerHandle[T], _ []T) ([]T, error) {
	return nil, utils.ErrBackendUnavailable
}

// UpdateWeights is not implemented in Phase 15; deferred to Phase 16.
func (b *openclBackend[T]) UpdateWeights(_ compute.LayerHandle[T], _, _ []T, _ T) error {
	return utils.ErrBackendUnavailable
}
