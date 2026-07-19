package network

import (
	"slices"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
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
			*h.GetValue() = activation.Activation(preact[cellIdx], act)
		}
		// Per-layer normalization: the next layer consumes the NORMALIZED
		// values, so the norm output is written back into the cells. The
		// pre-activation buffer keeps the un-normalized linear sum — σ′ in
		// the backward pass applies to the activation, not the norm output.
		if nl, ok := n.normLayers[i]; ok {
			x := make([]T, len(hb.cells))
			for j, h := range hb.cells {
				x[j] = *h.GetValue()
			}
			out := nl.Forward(x)
			for j, h := range hb.cells {
				*h.GetValue() = out[j]
			}
		}
		// Honest dropout: the mask gates the values the NEXT layer consumes
		// (audit B3: the old flat mask ran after the whole forward pass and
		// never influenced the output). Training passes only — inference
		// keeps every unit (REG-3).
		if n.training && n.masker != nil {
			x := make([]T, len(hb.cells))
			for j, h := range hb.cells {
				x[j] = *h.GetValue()
			}
			x = n.masker.MaskForwardLayer(i, x)
			for j, h := range hb.cells {
				*h.GetValue() = x[j]
			}
		}
	}
	// Output pre-activations first (needed whole-vector for softmax).
	for i, o := range n.Output.cells {
		o.Dense.CalculateValue()
		n.preactOutput[i] = *o.GetValue()
	}
	// Output activation: true vector softmax, else element-wise dispatch.
	if n.outputAct == activation.SOFTMAX {
		activation.SoftmaxInto(n.outActBuf, n.preactOutput)
		for i, o := range n.Output.cells {
			o.SetValue(n.outActBuf[i])
		}
	} else {
		for i, o := range n.Output.cells {
			o.SetValue(activation.Activation(n.preactOutput[i], n.outputAct))
		}
	}
	// Residual miss = −∂ℓ/∂y so backprop drives the CONFIGURED loss (not MSE
	// for everyone). Fused pairs use the (t − y) shortcut whose σ′ is folded
	// analytically; outputEffDeriv returns 1 for them.
	for _, o := range n.Output.cells {
		t := o.GetTarget()
		if t == nil {
			continue
		}
		y := *o.GetValue()
		if n.fusedOutput {
			o.SetMiss(*t - y)
		} else {
			o.SetMiss(-loss.Derivative(y, *t, n.lossMode))
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
	np := n.Output.Len()
	y := make([]T, np)
	tgt := make([]T, np)
	for i, o := range n.Output.cells {
		y[i] = *o.GetValue()
		if tp := o.GetTarget(); tp != nil {
			tgt[i] = *tp
		}
	}
	return loss.Aggregate(y, tgt, mode)
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
		// Aggregate the upstream error signal: for each source cell k in the
		// next layer, propagate its effective delta δ_k = σ′(preact_k)·miss_k
		// back through every axon k→j, accumulating δ_k·weight into miss_j.
		// Folding the SOURCE layer's activation derivative here is the chain
		// rule — omitting it (the historical bug) dropped one σ′ factor per
		// layer crossed and corrupted every multi-layer gradient. The type
		// filter to *cell.Hidden[T] excludes bias cells (present in *.Axons
		// but never receiving miss).
		if i == len(n.Hiddens)-1 {
			for oi, o := range n.Output.cells {
				delta := *o.GetMiss() * n.outputEffDeriv(oi)
				for _, a := range o.Axons {
					if h, ok := any(a.Cell).(*cell.Hidden[T]); ok {
						h.AddMiss(delta * a.Weight)
					}
				}
			}
		} else {
			nextAct := n.hiddenActs[i+1]
			nextPreact := n.preactHiddens[i+1]
			for ci, c := range n.Hiddens[i+1].cells {
				delta := *c.GetMiss() * activation.Derivative(nextPreact[ci], nextAct)
				for _, a := range c.Axons {
					if h, ok := any(a.Cell).(*cell.Hidden[T]); ok {
						h.AddMiss(delta * a.Weight)
					}
				}
			}
		}
		// Reverse of the forward's post-processing chain (norm → mask):
		// first undo the dropout gate — dropped units get zero miss,
		// retained units scale by 1/p. Elementwise scaling is sign-agnostic,
		// so the miss convention passes through unchanged.
		if n.training && n.masker != nil {
			up := make([]T, len(hb.cells))
			for j, h := range hb.cells {
				up[j] = *h.GetMiss()
			}
			up = n.masker.MaskBackwardLayer(i, up)
			for j, h := range hb.cells {
				h.SetMiss(up[j])
			}
		}
		// Route the accumulated miss through the layer's normalizer: what
		// arrived is −∂L/∂(norm output); downstream consumers (the next
		// boundary, CalculateWeights, AppendFlatGradients) need
		// −∂L/∂(activation). FeatureNorm.Backward works in the true-gradient
		// convention (it also accumulates ∂L/∂γ, ∂L/∂β), so the miss is
		// negated on the way in and the result negated on the way out.
		if nl, ok := n.normLayers[i]; ok {
			up := make([]T, len(hb.cells))
			for j, h := range hb.cells {
				up[j] = -*h.GetMiss()
			}
			dx := nl.Backward(up)
			for j, h := range hb.cells {
				h.SetMiss(-dx[j])
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
			eff := *rate * activation.Derivative(preact[cellIdx], act)
			h.CalculateWeight(&eff)
		}
	}
	for cellIdx, o := range n.Output.cells {
		eff := *rate * n.outputEffDeriv(cellIdx)
		o.CalculateWeight(&eff)
	}
}

// InferDense runs a stateless forward pass over the dense graph and returns a
// freshly allocated output slice. Unlike CalculateValues it never writes cell
// value or miss fields — every intermediate activation lives in a local map —
// so any number of goroutines may call it concurrently under a read lock
// without racing (audit D1: the old Query mutated shared cells and returned
// wrong answers under load). Weights and bias values are only read; the caller
// guarantees no concurrent Train via the facade's RWMutex.
//
// AI-Meta:
//   - Purpose: Read-only forward pass for concurrent inference; no shared cell mutation (bias values are read-only after Build).
//   - Concurrency: ReadSafe.
//   - Related: [CalculateValues], [nn.NN.Query].
func (n *Network[T]) InferDense(input []T) ([]T, error) {
	if len(input) != n.Input.Len() {
		return nil, utils.Newf(utils.ErrInputData,
			"InferDense: expected %d values, got %d", n.Input.Len(), len(input))
	}
	total := n.Input.Len()
	for _, hb := range n.Hiddens {
		total += hb.Len()
	}
	values := make(map[neuron.Nucleus[T]]T, total)
	for j, c := range n.Input.cells {
		if !isFinite(input[j]) {
			return nil, utils.Newf(utils.ErrInputData,
				"InferDense: non-finite input at index %d", j)
		}
		values[c] = input[j]
	}
	// sourceValue reads a cell's activation from the local map, falling back to
	// the cell's stored value for bias cells (immutable after Build, safe to
	// read concurrently).
	sourceValue := func(c neuron.Nucleus[T]) T {
		if v, ok := values[c]; ok {
			return v
		}
		return *c.GetValue()
	}
	for i, hb := range n.Hiddens {
		act := n.hiddenActs[i]
		for _, h := range hb.cells {
			var sum T
			for _, a := range h.Axons {
				sum += a.Weight * sourceValue(a.Cell)
			}
			values[h] = activation.Activation(sum, act)
		}
		// Normalization in the read-only path uses ForwardInference — pure by
		// contract (no running-stat updates, no caches) so concurrent Query
		// goroutines can share the normalizer instance.
		if nl, ok := n.normLayers[i]; ok {
			x := make([]T, len(hb.cells))
			for j, h := range hb.cells {
				x[j] = values[h]
			}
			out := nl.ForwardInference(x)
			for j, h := range hb.cells {
				values[h] = out[j]
			}
		}
	}
	preout := make([]T, n.Output.Len())
	for i, o := range n.Output.cells {
		var sum T
		for _, a := range o.Axons {
			sum += a.Weight * sourceValue(a.Cell)
		}
		preout[i] = sum
	}
	out := make([]T, n.Output.Len())
	if n.outputAct == activation.SOFTMAX {
		activation.SoftmaxInto(out, preout)
	} else {
		for i := range preout {
			out[i] = activation.Activation(preout[i], n.outputAct)
		}
	}
	return out, nil
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
			deriv := activation.Derivative(preact[cellIdx], act)
			miss := *h.GetMiss()
			for _, a := range h.Axons {
				dst = append(dst, -deriv*miss**a.Cell.GetValue())
			}
		}
	}
	for cellIdx, o := range n.Output.cells {
		deriv := n.outputEffDeriv(cellIdx)
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
		deriv := activation.Derivative(preact[cellIdx], act)
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
