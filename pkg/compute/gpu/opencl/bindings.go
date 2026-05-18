//go:build cgo && opencl

package opencl

// #cgo LDFLAGS: -lOpenCL
// #include <CL/cl.h>
// #include <stdlib.h>
//
// // clGetPlatformIDsSafe is a nil-safe wrapper so we can call it without
// // a pre-allocated platform array when we only want the count.
// static cl_int getPlatformCount(cl_uint *count) {
//     return clGetPlatformIDs(0, NULL, count);
// }
import "C"
