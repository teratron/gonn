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
