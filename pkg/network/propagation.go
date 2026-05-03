package network

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron/cell"
)

// CalculateValues runs the forward pass left-to-right across the chain:
// every hidden layer (in order) consumes the previous layer's outputs,
// then Output consumes the last hidden layer. Each cell's CalculateValue
// reads its own incoming Axons, so layer-i cells see exactly the values
// produced by layer i-1 cells in the same forward pass.
//
// After the linear sum is computed, it is captured in preactHiddens[i]
// / preactOutput (backprop needs pre-activation values for the
// derivative call), then the layer-wide activation function is applied
// via the [pkg/activation] dispatcher. Per [l2-multihidden-impl] §5.6.
func (n *Network[T]) CalculateValues() {
	for i, hb := range n.Hiddens {
		act := n.hiddenActs[i]
		preact := n.preactHiddens[i]
		for cellIdx, h := range hb.cells {
			h.CalculateValue()
			preact[cellIdx] = *h.GetValue()
			*h.GetValue() = activation.Activation[T](preact[cellIdx], act)
		}
	}
	for i, o := range n.Output.cells {
		o.Dense.CalculateValue()
		n.preactOutput[i] = *o.GetValue()
		*o.GetValue() = activation.Activation[T](n.preactOutput[i], n.outputAct)
		if t := o.GetTarget(); t != nil {
			o.SetMiss(*t - *o.GetValue())
		}
	}
}

// CalculateLoss computes the aggregate loss across Output cells using
// the supplied dispatcher mode. Reads the residual already set by
// Output.CalculateValue.
func (n *Network[T]) CalculateLoss(mode loss.Type) T {
	misses := make([]*T, n.Output.Len())
	for i, o := range n.Output.cells {
		misses[i] = o.GetMiss()
	}
	return loss.CalculateTotalLoss(&misses, mode)
}

// CalculateMisses runs the backward pass per [l2-multihidden-impl] §5.6.
// Output cells already hold the residual computed during forward
// (CalculateValues does `target - value`); we promote it to δ_o by
// multiplying by σ'(z_o), then walk Hiddens right-to-left, accumulating
//
//	δ_i = (Σ over next-layer cells c: c.miss × axon.Weight) × σ'(z_i)
//
// where axon.Cell points at the source cell in Hiddens[i]. Bias cells
// (also reachable through Axons) are filtered out by the type assertion
// to *cell.Hidden[T] — biases never accumulate gradient.
//
// For len(Hiddens) == 1 this reduces to the v0.1 single-hidden formula
// composed with the output derivative; XOR convergence is preserved.
func (n *Network[T]) CalculateMisses() {
	for i, o := range n.Output.cells {
		dz := activation.Derivative[T](n.preactOutput[i], n.outputAct)
		o.SetMiss(*o.GetMiss() * dz)
	}

	for i := len(n.Hiddens) - 1; i >= 0; i-- {
		hb := n.Hiddens[i]
		for _, h := range hb.cells {
			h.SetMiss(0)
		}
		// Aggregate raw upstream contribution: Σ next.miss × axon.weight
		// over each axon in the next layer pointing back to a cell in
		// Hiddens[i]. The type filter to *cell.Hidden[T] excludes bias
		// cells (which also appear in *.Axons but never receive miss).
		if i == len(n.Hiddens)-1 {
			for _, o := range n.Output.cells {
				ms := *o.GetMiss()
				for _, a := range o.Axons {
					if h, ok := any(a.Cell).(*cell.Hidden[T]); ok {
						h.AddMiss(ms * a.Weight)
					}
				}
			}
		} else {
			for _, c := range n.Hiddens[i+1].cells {
				ms := *c.GetMiss()
				for _, a := range c.Axons {
					if h, ok := any(a.Cell).(*cell.Hidden[T]); ok {
						h.AddMiss(ms * a.Weight)
					}
				}
			}
		}
		// Fold in this layer's activation derivative — δ_i = raw × σ'(z_i).
		act := n.hiddenActs[i]
		preact := n.preactHiddens[i]
		for cellIdx, h := range hb.cells {
			dz := activation.Derivative[T](preact[cellIdx], act)
			h.SetMiss(*h.GetMiss() * dz)
		}
	}
}

// CalculateWeights applies the gradient-descent step to every learnable
// cell — every Hidden layer plus Output. Per [l2-multihidden-impl] §5.6
// each cell's miss already carries σ'(z) (folded in by
// CalculateMisses), so the per-cell effective rate is simply `rate`:
// the axon update reduces to `weight += rate × cell.miss × cell.value()`.
//
// Single-hidden (len(Hiddens) == 1) reduces to one outer iteration over
// Hiddens[0] and Output — bit-for-bit equivalent to the v0.1 path
// modulo the moved-derivative bookkeeping.
func (n *Network[T]) CalculateWeights(rate *T) {
	for _, hb := range n.Hiddens {
		for _, h := range hb.cells {
			h.CalculateWeight(rate)
		}
	}
	for _, o := range n.Output.cells {
		o.CalculateWeight(rate)
	}
}

// CalculateLossDefault is a convenience wrapper that uses the loss mode
// captured by SetLayers — saves callers from re-deriving it each step.
func (n *Network[T]) CalculateLossDefault() T {
	return n.CalculateLoss(n.lossMode)
}

// Train runs one full forward + backward + weight-update step on the
// supplied (input, target) pair using the network's configured rate.
func (n *Network[T]) Train(input, target []T) (T, error) {
	if err := n.SetInputs(input); err != nil {
		return 0, err
	}
	if err := n.SetTargets(target); err != nil {
		return 0, err
	}
	n.CalculateValues()
	loss := n.CalculateLossDefault()
	n.CalculateMisses()
	rate := n.LearningRate
	n.CalculateWeights(&rate)
	return loss, nil
}

// _ keeps the cell import used by CalculateMisses.
var _ = (*cell.Hidden[float32])(nil)
