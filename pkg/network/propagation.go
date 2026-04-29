package network

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron/cell"
)

// CalculateValues runs the forward pass: Hidden first (consumes Input),
// then Output (consumes Hidden). Each cell's CalculateValue reads its
// own incoming Axons — no slice covariance needed. After the linear
// sum is computed, it is captured in preactHidden / preactOutput (the
// backprop pass needs pre-activation values for the derivative call),
// then the layer-wide activation function is applied via the
// [pkg/activation] dispatcher.
func (n *Network[T]) CalculateValues() {
	for i, h := range n.Hidden.cells {
		h.CalculateValue()
		n.preactHidden[i] = *h.GetValue()
		*h.GetValue() = activation.Activation[T](n.preactHidden[i], n.hiddenAct)
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

// CalculateMisses runs the backward pass. Output cells already hold the
// residual computed during forward (Output.CalculateValue does
// `target - value`); Hidden misses are accumulated by walking each
// Output cell's incoming axons and crediting the source Hidden cell with
// `output.miss * axon.weight`. The reverse-walk relies on axon.Cell
// (incoming endpoint), which closes [l2-network-graph] §5.4 #6 (the
// legacy code iterated Output.cells using Hidden.size, an off-by-one /
// shape-mismatch bug).
func (n *Network[T]) CalculateMisses() {
	for _, h := range n.Hidden.cells {
		h.SetMiss(0)
	}
	for _, o := range n.Output.cells {
		ms := *o.GetMiss()
		for _, a := range o.Axons {
			if h, ok := any(a.Cell).(*cell.Hidden[T]); ok {
				h.AddMiss(ms * a.Weight)
			}
		}
	}
}

// CalculateWeights applies the gradient-descent step to every learnable
// cell — Hidden and Output. rate is the supplied learning rate; the
// activation derivative for the cell's layer is folded into the
// effective rate so that downstream cell.CalculateWeight uses
// `rate * derivative * miss`. Pre-activation values captured during the
// forward pass are fed to the derivative dispatcher.
func (n *Network[T]) CalculateWeights(rate *T) {
	for i, h := range n.Hidden.cells {
		eff := *rate * activation.Derivative[T](n.preactHidden[i], n.hiddenAct)
		h.CalculateWeight(&eff)
	}
	for i, o := range n.Output.cells {
		eff := *rate * activation.Derivative[T](n.preactOutput[i], n.outputAct)
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
