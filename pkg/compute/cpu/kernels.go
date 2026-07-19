// Package cpu — CPU forward / backward / update kernels.
//
// Reference math implementations defined in [l2-backend-cpu] §5.2.
// Per COMP-1 these results are the "golden" output every alternative
// backend must match within ToleranceF32 / ToleranceF64.
package cpu

import (
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// Forward computes the layer's output activations from the input vector.
// out[i] = activation( sum_j(input[j] * weights[i][j]) + bias[i] ).
// The function returns a freshly allocated slice so callers can safely
// retain it; reuse the slice via Backend.UpdateWeights when avoiding
// allocations matters (PERF-2).
func (b Backend[T]) Forward(layer compute.LayerHandle[T], input []T) ([]T, error) {
	if layer.Size != len(layer.Weights) {
		return nil, utils.Newf(utils.ErrCompute,
			"cpu.Forward: layer.Size %d != len(weights) %d", layer.Size, len(layer.Weights))
	}
	if layer.Activation == nil {
		return nil, utils.Newf(utils.ErrCompute, "cpu.Forward: layer.Activation is nil")
	}
	out := make([]T, layer.Size)
	for i := range layer.Size {
		row := layer.Weights[i]
		if len(row) != len(input) {
			return nil, utils.Newf(utils.ErrCompute,
				"cpu.Forward: weights[%d] len %d != input len %d", i, len(row), len(input))
		}
		var sum T
		for j, w := range row {
			sum += input[j] * w
		}
		if i < len(layer.Bias) {
			sum += layer.Bias[i]
		}
		out[i] = layer.Activation(sum)
	}
	return out, nil
}

// Backward computes the gradient with respect to the layer's input
// given the gradient with respect to its output. dInput[j] =
// sum_i( gradient[i] * derivative(out[i]) * weights[i][j] ). The caller
// supplies derivative-applied gradients via layer.Derivative — keeping
// the kernel pure linear-algebra makes the math easy to verify.
func (b Backend[T]) Backward(layer compute.LayerHandle[T], gradient []T) ([]T, error) {
	if layer.Size != len(gradient) {
		return nil, utils.Newf(utils.ErrCompute,
			"cpu.Backward: gradient len %d != layer.Size %d", len(gradient), layer.Size)
	}
	if len(layer.Weights) == 0 {
		return nil, utils.Newf(utils.ErrCompute, "cpu.Backward: empty weights")
	}
	inSize := len(layer.Weights[0])
	dInput := make([]T, inSize)
	for i, g := range gradient {
		row := layer.Weights[i]
		if len(row) != inSize {
			return nil, utils.Newf(utils.ErrCompute,
				"cpu.Backward: ragged weights at row %d", i)
		}
		for j, w := range row {
			dInput[j] += g * w
		}
	}
	return dInput, nil
}

// Compile-time assertion that the CPU backend supplies the accelerated
// dense path. Because these are the reference implementations, a backend
// disagreeing with them is by definition the one that is wrong (COMP-1).
var (
	_ compute.DenseKernels[float32] = Backend[float32]{}
	_ compute.DenseKernels[float64] = Backend[float64]{}
)

// MatVec computes preact[o] = Σ_j W[o*In+j] * input[j].
//
// The row subslice is hoisted out of the inner loop so the bounds check is
// paid once per row rather than once per element — the single cheapest win
// available on contiguous storage in Go.
//
// AI-Meta:
//   - Purpose: Reference dense matrix-vector product for the forward pass.
//   - Errors: ErrCompute (shape mismatch).
//   - Concurrency: NotSafe.
//   - Related: [compute.DenseKernels], [Backend.MatVecT].
func (b Backend[T]) MatVec(m compute.DenseMatrix[T], input, preact []T) error {
	if err := checkMatrix(m, "MatVec"); err != nil {
		return err
	}
	if len(input) != m.In || len(preact) != m.Out {
		return utils.Newf(utils.ErrCompute,
			"cpu.MatVec: input len %d (want %d), preact len %d (want %d)",
			len(input), m.In, len(preact), m.Out)
	}
	for o := range m.Out {
		row := m.W[o*m.In : (o+1)*m.In : (o+1)*m.In]
		var sum T
		for j, w := range row {
			sum += w * input[j]
		}
		preact[o] = sum
	}
	return nil
}

// MatVecT computes dInput[j] = Σ_o delta[o] * W[o*In+j].
//
// Iteration is row-major (outer over o, inner over j) even though the result
// is indexed by j: walking W in storage order beats the "natural" column loop,
// which would stride by In on every step.
//
// AI-Meta:
//   - Purpose: Reference transposed matrix-vector product for the backward pass.
//   - Errors: ErrCompute (shape mismatch).
//   - Concurrency: NotSafe.
//   - Related: [compute.DenseKernels], [Backend.MatVec].
func (b Backend[T]) MatVecT(m compute.DenseMatrix[T], delta, dInput []T) error {
	if err := checkMatrix(m, "MatVecT"); err != nil {
		return err
	}
	if len(delta) != m.Out || len(dInput) != m.In {
		return utils.Newf(utils.ErrCompute,
			"cpu.MatVecT: delta len %d (want %d), dInput len %d (want %d)",
			len(delta), m.Out, len(dInput), m.In)
	}
	clear(dInput)
	for o := range m.Out {
		d := delta[o]
		if d == 0 {
			continue
		}
		row := m.W[o*m.In : (o+1)*m.In : (o+1)*m.In]
		for j, w := range row {
			dInput[j] += d * w
		}
	}
	return nil
}

// GradOuter computes grad[o*In+j] = scale * delta[o] * input[j].
//
// AI-Meta:
//   - Purpose: Reference outer-product gradient for one sample.
//   - Errors: ErrCompute (shape mismatch).
//   - Concurrency: NotSafe.
//   - Related: [compute.DenseKernels].
func (b Backend[T]) GradOuter(m compute.DenseMatrix[T], delta, input, grad []T, scale T) error {
	if err := checkMatrix(m, "GradOuter"); err != nil {
		return err
	}
	if len(delta) != m.Out || len(input) != m.In || len(grad) != m.In*m.Out {
		return utils.Newf(utils.ErrCompute,
			"cpu.GradOuter: delta len %d (want %d), input len %d (want %d), grad len %d (want %d)",
			len(delta), m.Out, len(input), m.In, len(grad), m.In*m.Out)
	}
	for o := range m.Out {
		coef := scale * delta[o]
		row := grad[o*m.In : (o+1)*m.In : (o+1)*m.In]
		for j := range row {
			row[j] = coef * input[j]
		}
	}
	return nil
}

// checkMatrix validates the shape contract shared by all three kernels.
func checkMatrix[T utils.Float](m compute.DenseMatrix[T], op string) error {
	if m.In <= 0 || m.Out <= 0 {
		return utils.Newf(utils.ErrCompute, "cpu.%s: degenerate shape In=%d Out=%d", op, m.In, m.Out)
	}
	if len(m.W) != m.In*m.Out {
		return utils.Newf(utils.ErrCompute,
			"cpu.%s: weight len %d != In*Out (%d*%d)", op, len(m.W), m.In, m.Out)
	}
	return nil
}

// UpdateWeights applies a vanilla SGD step to the layer's weights and
// bias. weights[i][j] -= rate * deltas[i] * inputs[j]; bias[i] -=
// rate * deltas[i]. The function mutates layer.Weights / layer.Bias in
// place — callers must serialise concurrent UpdateWeights calls on the
// same layer.
func (b Backend[T]) UpdateWeights(layer compute.LayerHandle[T], inputs, deltas []T, rate T) error {
	if len(deltas) != layer.Size {
		return utils.Newf(utils.ErrCompute,
			"cpu.UpdateWeights: deltas len %d != layer.Size %d", len(deltas), layer.Size)
	}
	for i := range layer.Size {
		row := layer.Weights[i]
		if len(row) != len(inputs) {
			return utils.Newf(utils.ErrCompute,
				"cpu.UpdateWeights: weights[%d] len %d != inputs len %d", i, len(row), len(inputs))
		}
		step := rate * deltas[i]
		for j := range row {
			row[j] -= step * inputs[j]
		}
		if i < len(layer.Bias) {
			layer.Bias[i] -= step
		}
	}
	return nil
}
