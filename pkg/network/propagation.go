package network

import (
	"slices"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron/cell"
)

// CalculateValues runs the forward pass left-to-right: each hidden layer
// consumes the previous layer's post-activation outputs; Output consumes
// the last hidden layer. Pre-activation linear sums are captured before
// the activation dispatcher is applied so CalculateMisses/CalculateWeights
// can feed the correct value to the derivative.
//
// AI-Meta:
//   - Purpose: Execute the full forward pass, writing post-activation values and miss residuals into cells.
//   - Concurrency: NotSafe; mutates cell values and preact scratch buffers.
//   - Related: [CalculateMisses], [CalculateWeights], [Train].
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

// CalculateLoss computes the aggregate loss across Output cells using the
// supplied loss mode. Reads residuals written by CalculateValues.
//
// AI-Meta:
//   - Purpose: Compute scalar training loss after a forward pass; useful for logging or early stopping.
//   - Concurrency: ReadSafe.
//   - Related: [CalculateLossDefault], [CalculateValues], [loss.CalculateTotalLoss].
func (n *Network[T]) CalculateLoss(mode loss.Type) T {
	misses := make([]*T, n.Output.Len())
	for i, o := range n.Output.cells {
		misses[i] = o.GetMiss()
	}
	return loss.CalculateTotalLoss(&misses, mode)
}

// CalculateMisses runs the backward pass right-to-left across the hidden
// chain. Output cell residuals are already set during CalculateValues.
// Each hidden cell accumulates raw miss = Σ(next.miss × axon.Weight);
// the activation derivative is NOT folded here — CalculateWeights folds
// it into the effective rate per layer. Bias cells are filtered by type
// assertion and never receive gradient.
//
// AI-Meta:
//   - Purpose: Propagate error signals backward through all hidden layers.
//   - Concurrency: NotSafe; mutates cell miss fields.
//   - Related: [CalculateValues], [CalculateWeights], [Train].
func (n *Network[T]) CalculateMisses() {
	for i, hb := range slices.Backward(n.Hiddens) {

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

// CalculateWeights applies one gradient-descent step to all learnable cells
// (Hidden layers + Output). The activation derivative is folded into the
// effective rate per cell: eff = rate × σ'(preact), so cell.CalculateWeight
// computes ΔW = eff × miss × axon.value without knowing the activation type.
//
// AI-Meta:
//   - Purpose: Update all axon weights from the current miss and pre-activation values.
//   - Concurrency: NotSafe; mutates axon weights in place.
//   - Related: [CalculateMisses], [CalculateValues], [Train].
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

// CalculateLossDefault calls CalculateLoss with the mode captured during
// SetLayers, avoiding the need to re-supply the loss type each step.
//
// AI-Meta:
//   - Purpose: Compute loss using the layer-configured mode; convenience wrapper around CalculateLoss.
//   - Concurrency: ReadSafe.
//   - Related: [CalculateLoss], [LossMode].
func (n *Network[T]) CalculateLossDefault() T {
	return n.CalculateLoss(n.lossMode)
}

// AppendFlatGradients computes the raw gradient ∂L/∂w for every learnable
// weight and appends it to dst (reusing the backing array when capacity
// allows). Gradients are emitted in the same canonical order as
// AppendFlatWeights: Hiddens[0] → Hiddens[n-1] → Output.
//
// Definition: grad[i] = −σ'(preact) × miss × axon.Cell.Value
// The sign convention ensures that SGD.Step (w -= lr × grad) reproduces
// the original inline update (w += lr × σ'(preact) × miss × cell.value).
//
// Must be called after CalculateMisses — it reads the miss fields set there.
//
// AI-Meta:
//   - Purpose: Compute flat raw gradients for the optimizer Step call.
//   - Concurrency: NotSafe; reads cell miss and pre-activation buffers.
//   - Related: [AppendFlatWeights], [ApplyFlatWeights], [CalculateMisses].
//   - Stability: Stable.
func (n *Network[T]) AppendFlatGradients(dst []T) []T {
	dst = dst[:0]
	for layerIdx, hb := range n.Hiddens {
		act := n.hiddenActs[layerIdx]
		preact := n.preactHiddens[layerIdx]
		for cellIdx, h := range hb.cells {
			deriv := activation.Derivative[T](preact[cellIdx], act)
			miss := *h.GetMiss()
			for _, a := range h.Axons {
				dst = append(dst, -deriv*miss**a.Cell.GetValue())
			}
		}
	}
	for cellIdx, o := range n.Output.cells {
		deriv := activation.Derivative[T](n.preactOutput[cellIdx], n.outputAct)
		miss := *o.GetMiss()
		for _, a := range o.Axons {
			dst = append(dst, -deriv*miss**a.Cell.GetValue())
		}
	}
	return dst
}

// AppendInputGradient computes ∂L/∂(input cell value) for every Input cell
// and appends the values into dst (reusing its backing array when capacity
// allows). Must be called AFTER [CalculateMisses] — it reads miss and the
// first hidden layer's pre-activation buffer.
//
// For input cell i the formula is:
//
//	∂L/∂x_i = − Σ_{j in Hiddens[0]}  σ'(preact_j) · miss_j · axon[i→j].weight
//
// The sign convention matches [AppendFlatGradients]: SGD applies
// w -= lr · grad, so the negation here reproduces the existing inline
// update w += lr · σ'(preact) · miss · axon.value for an upstream pre-stage
// (e.g. a conv prefix) that needs ∂L/∂x to drive its own Backward pass.
//
// AI-Meta:
//   - Purpose: Expose the input-cell gradient so an upstream stage (conv prefix) can run its backward pass.
//   - Usage: gradX := n.AppendInputGradient(gradX[:0]).
//   - Concurrency: NotSafe; must run after CalculateMisses, before CalculateWeights overwrites internal state.
//   - Related: [CalculateMisses], [AppendFlatGradients], [SetInputs].
//   - Stability: Stable.
func (n *Network[T]) AppendInputGradient(dst []T) []T {
	dst = dst[:0]
	inLen := n.Input.Len()
	if inLen == 0 || len(n.Hiddens) == 0 {
		return dst
	}
	act := n.hiddenActs[0]
	preact := n.preactHiddens[0]
	// Build a map from input-cell identity → index, so the axon lookup is
	// O(1) per axon. The Input bundle is small (matches the conv stack
	// output), so the map cost is negligible.
	inputIdx := make(map[any]int, inLen)
	for i, c := range n.Input.cells {
		inputIdx[any(c)] = i
	}
	dst = append(dst, make([]T, inLen)...)
	for cellIdx, h := range n.Hiddens[0].cells {
		deriv := activation.Derivative[T](preact[cellIdx], act)
		miss := *h.GetMiss()
		coef := deriv * miss
		for _, a := range h.Axons {
			if i, ok := inputIdx[any(a.Cell)]; ok {
				dst[i] -= coef * a.Weight
			}
		}
	}
	return dst
}

// Train runs one full forward + backward + weight-update step on the
// supplied (input, target) pair using the network's LearningRate.
//
// AI-Meta:
//   - Purpose: Execute one training step; returns the aggregate loss for the sample.
//   - Errors: ErrInputData (length mismatch in SetInputs or SetTargets).
//   - Concurrency: NotSafe; the single-step contract requires exclusive access.
//   - Related: [CalculateValues], [CalculateMisses], [CalculateWeights], [SetInputs], [SetTargets].
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
