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
	kernelGradWF32 = "dense_grad_w_f32"
	kernelGradXF32 = "dense_grad_x_f32"
	kernelGradWF64 = "dense_grad_w_f64"
	kernelGradXF64 = "dense_grad_x_f64"
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

// kernelDenseBackward launches the dense_grad_w + dense_grad_x kernel pair for T.
// Returns (gradW, gradX) or an error. b.program must be non-nil.
//
// gradW = dY ⊗ input  (∂L/∂W)   shape: outSize * inSize
// gradX = W^T · dY    (∂L/∂X)   shape: inSize
func kernelDenseBackward[T utils.Float](b *openclBackend[T], layer compute.LayerHandle[T], input, dY []T) ([]T, []T, error) {
	if b.program == nil {
		prog, err := buildProgram(b.ctx)
		if err != nil {
			return nil, nil, err
		}
		b.program = prog
	}

	outSize := len(dY)
	inSize := len(input)

	var zero T
	var nameGW, nameGX string
	if unsafe.Sizeof(zero) == 4 {
		nameGW, nameGX = kernelGradWF32, kernelGradXF32
	} else {
		nameGW, nameGX = kernelGradWF64, kernelGradXF64
	}

	// Flatten weights for gradX computation.
	weights := make([]T, outSize*inSize)
	for i, row := range layer.Weights {
		copy(weights[i*inSize:], row)
	}

	// Allocate device buffers.
	dYBuf, err := allocBuf[T](b.reg, b.ctx, outSize)
	if err != nil {
		return nil, nil, err
	}
	defer freeBuf[T](b.reg, dYBuf)

	inBuf, err := allocBuf[T](b.reg, b.ctx, inSize)
	if err != nil {
		return nil, nil, err
	}
	defer freeBuf[T](b.reg, inBuf)

	wBuf, err := allocBuf[T](b.reg, b.ctx, outSize*inSize)
	if err != nil {
		return nil, nil, err
	}
	defer freeBuf[T](b.reg, wBuf)

	gwBuf, err := allocBuf[T](b.reg, b.ctx, outSize*inSize)
	if err != nil {
		return nil, nil, err
	}
	defer freeBuf[T](b.reg, gwBuf)

	gxBuf, err := allocBuf[T](b.reg, b.ctx, inSize)
	if err != nil {
		return nil, nil, err
	}
	defer freeBuf[T](b.reg, gxBuf)

	if err := writeBuf[T](b.reg, b.queue, dYBuf, dY); err != nil {
		return nil, nil, err
	}
	if err := writeBuf[T](b.reg, b.queue, inBuf, input); err != nil {
		return nil, nil, err
	}
	if err := writeBuf[T](b.reg, b.queue, wBuf, weights); err != nil {
		return nil, nil, err
	}

	setArg := func(k C.cl_kernel, idx C.cl_uint, obj C.cl_mem) error {
		if ret := C.clSetKernelArg(k, idx, C.size_t(unsafe.Sizeof(obj)), unsafe.Pointer(&obj)); ret != C.CL_SUCCESS {
			return utils.Newf(utils.ErrBackendKernel, "opencl: clSetKernelArg(%d) failed (%d)", idx, int(ret))
		}
		return nil
	}
	setIntArg := func(k C.cl_kernel, idx C.cl_uint, v C.int) error {
		if ret := C.clSetKernelArg(k, idx, C.size_t(unsafe.Sizeof(v)), unsafe.Pointer(&v)); ret != C.CL_SUCCESS {
			return utils.Newf(utils.ErrBackendKernel, "opencl: clSetKernelArg int(%d) failed (%d)", idx, int(ret))
		}
		return nil
	}

	dYEntry, _ := lookupBuf[T](b.reg, dYBuf)
	inEntry, _ := lookupBuf[T](b.reg, inBuf)
	wEntry, _ := lookupBuf[T](b.reg, wBuf)
	gwEntry, _ := lookupBuf[T](b.reg, gwBuf)
	gxEntry, _ := lookupBuf[T](b.reg, gxBuf)

	// Launch dense_grad_w: global size = outSize * inSize.
	{
		cname := C.CString(nameGW)
		defer C.free(unsafe.Pointer(cname))
		var errCode C.cl_int
		k := C.clCreateKernel(b.program, cname, &errCode)
		if errCode != C.CL_SUCCESS {
			return nil, nil, utils.Newf(utils.ErrBackendKernel, "opencl: clCreateKernel(%s) failed (%d)", nameGW, int(errCode))
		}
		defer C.clReleaseKernel(k)
		if err := setArg(k, 0, dYEntry.mem); err != nil {
			return nil, nil, err
		}
		if err := setArg(k, 1, inEntry.mem); err != nil {
			return nil, nil, err
		}
		if err := setArg(k, 2, gwEntry.mem); err != nil {
			return nil, nil, err
		}
		if err := setIntArg(k, 3, C.int(inSize)); err != nil {
			return nil, nil, err
		}
		if err := setIntArg(k, 4, C.int(outSize)); err != nil {
			return nil, nil, err
		}
		gs := C.size_t(outSize * inSize)
		if ret := C.clEnqueueNDRangeKernel(b.queue, k, 1, nil, &gs, nil, 0, nil, nil); ret != C.CL_SUCCESS {
			return nil, nil, utils.Newf(utils.ErrBackendKernel, "opencl: dense_grad_w NDRange failed (%d)", int(ret))
		}
	}

	// Launch dense_grad_x: global size = inSize.
	{
		cname := C.CString(nameGX)
		defer C.free(unsafe.Pointer(cname))
		var errCode C.cl_int
		k := C.clCreateKernel(b.program, cname, &errCode)
		if errCode != C.CL_SUCCESS {
			return nil, nil, utils.Newf(utils.ErrBackendKernel, "opencl: clCreateKernel(%s) failed (%d)", nameGX, int(errCode))
		}
		defer C.clReleaseKernel(k)
		if err := setArg(k, 0, wEntry.mem); err != nil {
			return nil, nil, err
		}
		if err := setArg(k, 1, dYEntry.mem); err != nil {
			return nil, nil, err
		}
		if err := setArg(k, 2, gxEntry.mem); err != nil {
			return nil, nil, err
		}
		if err := setIntArg(k, 3, C.int(inSize)); err != nil {
			return nil, nil, err
		}
		if err := setIntArg(k, 4, C.int(outSize)); err != nil {
			return nil, nil, err
		}
		gs := C.size_t(inSize)
		if ret := C.clEnqueueNDRangeKernel(b.queue, k, 1, nil, &gs, nil, 0, nil, nil); ret != C.CL_SUCCESS {
			return nil, nil, utils.Newf(utils.ErrBackendKernel, "opencl: dense_grad_x NDRange failed (%d)", int(ret))
		}
	}

	if ret := C.clFinish(b.queue); ret != C.CL_SUCCESS {
		return nil, nil, utils.Newf(utils.ErrBackendKernel, "opencl: clFinish (backward) failed (%d)", int(ret))
	}

	gradW := make([]T, outSize*inSize)
	if err := readBuf[T](b.reg, b.queue, gwBuf, gradW); err != nil {
		return nil, nil, err
	}
	gradX := make([]T, inSize)
	if err := readBuf[T](b.reg, b.queue, gxBuf, gradX); err != nil {
		return nil, nil, err
	}
	return gradW, gradX, nil
}
