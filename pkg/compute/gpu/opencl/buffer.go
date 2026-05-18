//go:build cgo && opencl

package opencl

// #include <CL/cl.h>
import "C"

import (
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// gpuBuf holds an OpenCL device-memory handle and its logical element count.
type gpuBuf struct {
	mem    C.cl_mem
	length int
}

// bufferRegistry maps opaque uint64 IDs to device-side gpuBuf entries.
// Each openclBackend embeds one registry.
type bufferRegistry struct {
	mu      sync.Mutex
	entries map[uint64]gpuBuf
	nextID  atomic.Uint64
}

func newRegistry() *bufferRegistry {
	return &bufferRegistry{entries: make(map[uint64]gpuBuf)}
}

// encodeID stores id into the first 8 bytes of data (works for float32 with
// 2+ elements, and float64 with 1+ element).
func encodeID[T utils.Float](data []T, id uint64) {
	*(*uint64)(unsafe.Pointer(&data[0])) = id
}

func decodeID[T utils.Float](data []T) uint64 {
	return *(*uint64)(unsafe.Pointer(&data[0]))
}

// elemPad returns the minimum number of T elements needed to hold 8 bytes.
func elemPad[T utils.Float]() int {
	var zero T
	size := int(unsafe.Sizeof(zero))
	if size >= 8 {
		return 1
	}
	return (8 + size - 1) / size
}

// allocBuf creates a cl_mem device buffer of size T-elements, stores it in
// the registry, and returns a compute.Buffer[T] whose Data encodes the opaque ID.
func allocBuf[T utils.Float](r *bufferRegistry, ctx C.cl_context, size int) (compute.Buffer[T], error) {
	if ctx == nil {
		return compute.Buffer[T]{}, utils.ErrBackendUnavailable
	}
	var zero T
	elemBytes := C.size_t(unsafe.Sizeof(zero))
	var errCode C.cl_int
	mem := C.clCreateBuffer(ctx,
		C.CL_MEM_READ_WRITE,
		elemBytes*C.size_t(size),
		nil,
		&errCode,
	)
	if errCode != C.CL_SUCCESS {
		return compute.Buffer[T]{}, utils.Newf(utils.ErrBackendKernel,
			"opencl: clCreateBuffer failed (%d)", int(errCode))
	}

	id := r.nextID.Add(1)
	r.mu.Lock()
	r.entries[id] = gpuBuf{mem: mem, length: size}
	r.mu.Unlock()

	// Encode id into Data; pad ensures uint64 fits regardless of sizeof(T).
	data := make([]T, elemPad[T]())
	encodeID(data, id)
	return compute.Buffer[T]{Data: data}, nil
}

// freeBuf releases the cl_mem for the given buffer and removes it from the registry.
func freeBuf[T utils.Float](r *bufferRegistry, buf compute.Buffer[T]) error {
	if len(buf.Data) == 0 {
		return nil
	}
	id := decodeID(buf.Data)
	r.mu.Lock()
	entry, ok := r.entries[id]
	if ok {
		delete(r.entries, id)
	}
	r.mu.Unlock()
	if !ok {
		return nil
	}
	if ret := C.clReleaseMemObject(entry.mem); ret != C.CL_SUCCESS {
		return utils.Newf(utils.ErrBackendKernel,
			"opencl: clReleaseMemObject failed (%d)", int(ret))
	}
	return nil
}

// lookupBuf returns the gpuBuf for buf, or an error if the id is unknown.
func lookupBuf[T utils.Float](r *bufferRegistry, buf compute.Buffer[T]) (gpuBuf, error) {
	if len(buf.Data) == 0 {
		return gpuBuf{}, utils.ErrBackendUnavailable
	}
	id := decodeID(buf.Data)
	r.mu.Lock()
	entry, ok := r.entries[id]
	r.mu.Unlock()
	if !ok {
		return gpuBuf{}, utils.Newf(utils.ErrBackendKernel,
			"opencl: unknown buffer id %d", id)
	}
	return entry, nil
}

// writeBuf transfers src from host to the device buffer (blocking).
func writeBuf[T utils.Float](r *bufferRegistry, queue C.cl_command_queue, buf compute.Buffer[T], src []T) error {
	entry, err := lookupBuf(r, buf)
	if err != nil {
		return err
	}
	if len(src) != entry.length {
		return utils.Newf(utils.ErrBackendTransfer,
			"opencl: writeBuf size mismatch: got %d, want %d", len(src), entry.length)
	}
	var zero T
	elemBytes := C.size_t(unsafe.Sizeof(zero))
	ret := C.clEnqueueWriteBuffer(
		queue, entry.mem,
		C.CL_TRUE, // blocking
		0,
		elemBytes*C.size_t(entry.length),
		unsafe.Pointer(&src[0]),
		0, nil, nil,
	)
	if ret != C.CL_SUCCESS {
		return utils.Newf(utils.ErrBackendTransfer,
			"opencl: clEnqueueWriteBuffer failed (%d)", int(ret))
	}
	return nil
}

// readBuf transfers device buffer contents to dst on the host (blocking).
func readBuf[T utils.Float](r *bufferRegistry, queue C.cl_command_queue, buf compute.Buffer[T], dst []T) error {
	entry, err := lookupBuf(r, buf)
	if err != nil {
		return err
	}
	if len(dst) != entry.length {
		return utils.Newf(utils.ErrBackendTransfer,
			"opencl: readBuf size mismatch: got %d, want %d", len(dst), entry.length)
	}
	var zero T
	elemBytes := C.size_t(unsafe.Sizeof(zero))
	ret := C.clEnqueueReadBuffer(
		queue, entry.mem,
		C.CL_TRUE, // blocking
		0,
		elemBytes*C.size_t(entry.length),
		unsafe.Pointer(&dst[0]),
		0, nil, nil,
	)
	if ret != C.CL_SUCCESS {
		return utils.Newf(utils.ErrBackendTransfer,
			"opencl: clEnqueueReadBuffer failed (%d)", int(ret))
	}
	return nil
}
