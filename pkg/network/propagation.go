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

// CalculateMisses runs the backward pass across the chain. Per
// [l2-multihidden-impl] §5.6 the residual on each Output cell is set
// during forward (`target - value`); we walk Hiddens right-to-left,
// accumulating each cell's raw miss as
//
//	miss_i = Σ over next-layer cells c: c.miss × axon.Weight
//
// where axon.Cell points at the source cell in Hiddens[i]. The
// activation derivative is NOT folded here — CalculateWeights composes
// it into the per-layer effective rate (`rate × σ'(z_i)`) so the
// single-hidden path stays bit-identical to v0.1: same residuals on
// Output, same per-cell raw misses on Hidden, same ΔW arithmetic.
// Bias cells (also reachable through Axons) are filtered out by the
// type assertion to *cell.Hidden[T] — biases never accumulate gradient.
func (n *Network[T]) CalculateMisses() {
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
	}
}

// CalculateWeights applies the gradient-descent step to every learnable
// cell — every Hidden layer plus Output. Per cell the activation
// derivative is folded into the effective rate so that downstream
// cell.CalculateWeight uses `rate × σ'(z) × miss × axon.cell.value()`
// — the v0.1 single-hidden formula extended positionally to every
// chain entry. Pre-activation values captured during the forward pass
// are fed to the derivative dispatcher.
func (n *Network[T]) CalculateWeights(rate *T) {
	for layerIdx, hb := range n.Hiddens {
		act := n.hiddenActs[layerIdx]
		preact := n.preactHiddens[layerIdx]
		for cellIdx, h := range hb.cells {
			eff := *rate * activation.Derivative[T](preact[cellIdx], act)
			h.CalculateWeight(&eff)
		}
	}
	for cellIdx, o := range n.Output.cells {
		eff := *rate * activation.Derivative[T](n.preactOutput[cellIdx], n.outputAct)
		o.CalculateWeight(&eff)
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
