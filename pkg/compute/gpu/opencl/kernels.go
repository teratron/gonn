//go:build cgo && opencl

package opencl

// #include <CL/cl.h>
// #include <stdlib.h>
import "C"

import (
	_ "embed"
	"unsafe"

	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

//go:embed kernels.cl
var kernelSource string

const (
	kernelDenseF32 = "dense_forward_f32"
	kernelDenseF64 = "dense_forward_f64"
)

// buildProgram compiles kernels.cl on the given context and device.
// Called lazily on first Forward invocation.
func buildProgram(ctx C.cl_context) (C.cl_program, error) {
	src := C.CString(kernelSource)
	defer C.free(unsafe.Pointer(src))

	var errCode C.cl_int
	prog := C.clCreateProgramWithSource(ctx, 1, &src, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, utils.Newf(utils.ErrBackendKernel,
			"opencl: clCreateProgramWithSource failed (%d)", int(errCode))
	}

	if ret := C.clBuildProgram(prog, 0, nil, nil, nil, nil); ret != C.CL_SUCCESS {
		C.clReleaseProgram(prog)
		return nil, utils.Newf(utils.ErrBackendKernel,
			"opencl: clBuildProgram failed (%d)", int(ret))
	}
	return prog, nil
}

// kernelDenseForward launches the appropriate dense_forward kernel for T.
// b.program must be non-nil (checked by the caller).
func kernelDenseForward[T utils.Float](b *openclBackend[T], layer compute.LayerHandle[T], input []T) ([]T, error) {
	// Ensure program is compiled.
	if b.program == nil {
		prog, err := buildProgram(b.ctx)
		if err != nil {
			return nil, err
		}
		b.program = prog
	}

	outSize := len(layer.Bias)
	inSize := len(input)

	// Select kernel name by type width.
	var zero T
	var name string
	if unsafe.Sizeof(zero) == 4 {
		name = kernelDenseF32
	} else {
		name = kernelDenseF64
	}

	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	var errCode C.cl_int
	kernel := C.clCreateKernel(b.program, cname, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, utils.Newf(utils.ErrBackendKernel,
			"opencl: clCreateKernel(%s) failed (%d)", name, int(errCode))
	}
	defer C.clReleaseKernel(kernel)

	// Flatten weights: layer.Weights is [outSize][inSize].
	weights := make([]T, outSize*inSize)
	for i, row := range layer.Weights {
		copy(weights[i*inSize:], row)
	}

	// Allocate device buffers.
	wBuf, err := allocBuf[T](b.reg, b.ctx, len(weights))
	if err != nil {
		return nil, err
	}
	defer freeBuf[T](b.reg, wBuf)

	biasBuf, err := allocBuf[T](b.reg, b.ctx, outSize)
	if err != nil {
		return nil, err
	}
	defer freeBuf[T](b.reg, biasBuf)

	inBuf, err := allocBuf[T](b.reg, b.ctx, inSize)
	if err != nil {
		return nil, err
	}
	defer freeBuf[T](b.reg, inBuf)

	outBuf, err := allocBuf[T](b.reg, b.ctx, outSize)
	if err != nil {
		return nil, err
	}
	defer freeBuf[T](b.reg, outBuf)

	// Upload inputs to device.
	if err := writeBuf[T](b.reg, b.queue, wBuf, weights); err != nil {
		return nil, err
	}
	if err := writeBuf[T](b.reg, b.queue, biasBuf, layer.Bias); err != nil {
		return nil, err
	}
	if err := writeBuf[T](b.reg, b.queue, inBuf, input); err != nil {
		return nil, err
	}

	// Set kernel arguments.
	setArg := func(idx C.cl_uint, obj C.cl_mem) error {
		if ret := C.clSetKernelArg(kernel, idx, C.size_t(unsafe.Sizeof(obj)), unsafe.Pointer(&obj)); ret != C.CL_SUCCESS {
			return utils.Newf(utils.ErrBackendKernel,
				"opencl: clSetKernelArg(%d) failed (%d)", idx, int(ret))
		}
		return nil
	}

	wEntry, _ := lookupBuf[T](b.reg, wBuf)
	bEntry, _ := lookupBuf[T](b.reg, biasBuf)
	iEntry, _ := lookupBuf[T](b.reg, inBuf)
	oEntry, _ := lookupBuf[T](b.reg, outBuf)

	if err := setArg(0, wEntry.mem); err != nil {
		return nil, err
	}
	if err := setArg(1, bEntry.mem); err != nil {
		return nil, err
	}
	if err := setArg(2, iEntry.mem); err != nil {
		return nil, err
	}
	if err := setArg(3, oEntry.mem); err != nil {
		return nil, err
	}

	inSizeC := C.int(inSize)
	outSizeC := C.int(outSize)
	if ret := C.clSetKernelArg(kernel, 4, C.size_t(unsafe.Sizeof(inSizeC)), unsafe.Pointer(&inSizeC)); ret != C.CL_SUCCESS {
		return nil, utils.Newf(utils.ErrBackendKernel, "opencl: clSetKernelArg(4) failed (%d)", int(ret))
	}
	if ret := C.clSetKernelArg(kernel, 5, C.size_t(unsafe.Sizeof(outSizeC)), unsafe.Pointer(&outSizeC)); ret != C.CL_SUCCESS {
		return nil, utils.Newf(utils.ErrBackendKernel, "opencl: clSetKernelArg(5) failed (%d)", int(ret))
	}

	// Enqueue kernel: one work-item per output neuron.
	globalSize := C.size_t(outSize)
	if ret := C.clEnqueueNDRangeKernel(b.queue, kernel, 1, nil, &globalSize, nil, 0, nil, nil); ret != C.CL_SUCCESS {
		return nil, utils.Newf(utils.ErrBackendKernel,
			"opencl: clEnqueueNDRangeKernel failed (%d)", int(ret))
	}

	// Blocking finish + read back.
	if ret := C.clFinish(b.queue); ret != C.CL_SUCCESS {
		return nil, utils.Newf(utils.ErrBackendKernel,
			"opencl: clFinish failed (%d)", int(ret))
	}

	output := make([]T, outSize)
	if err := readBuf[T](b.reg, b.queue, outBuf, output); err != nil {
		return nil, err
	}
	return output, nil
}
