package network

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// matrix returns the contiguous view of dl handed to the compute kernels.
func (dl *denseLayer[T]) matrix() compute.DenseMatrix[T] {
	return compute.DenseMatrix[T]{W: dl.store.W, In: dl.in, Out: dl.out}
}

// kernelFailure disables the accelerated path after a backend kernel reports an
// error. Shapes are fixed at Build time, so a failure here is a backend bug, not
// a user error: the engine says so once and falls back to the reference loops
// rather than training on whatever the broken kernel produced.
func (n *Network[T]) kernelFailure(op string, err error) {
	utils.Logger.Warn("compute kernel failed; falling back to the internal reference path",
		"op", op, "err", err.Error())
	n.kernels = nil
}

// matVec computes dl.preact = W · dl.input.
func (n *Network[T]) matVec(dl *denseLayer[T]) {
	if n.kernels != nil {
		if err := n.kernels.MatVec(dl.matrix(), dl.input, dl.preact); err != nil {
			n.kernelFailure("MatVec", err)
		} else {
			return
		}
	}
	refMatVec(dl.store.W, dl.in, dl.out, dl.input, dl.preact)
}

// matVecT computes dl.dInput = Wᵀ · dl.delta.
func (n *Network[T]) matVecT(dl *denseLayer[T]) {
	if n.kernels != nil {
		if err := n.kernels.MatVecT(dl.matrix(), dl.delta, dl.dInput); err != nil {
			n.kernelFailure("MatVecT", err)
		} else {
			return
		}
	}
	refMatVecT(dl.store.W, dl.in, dl.out, dl.delta, dl.dInput)
}

// gradOuter writes scale · delta ⊗ input into dst (length in*out).
func (n *Network[T]) gradOuter(dl *denseLayer[T], dst []T, scale T) {
	if n.kernels != nil {
		if err := n.kernels.GradOuter(dl.matrix(), dl.delta, dl.input, dst, scale); err != nil {
			n.kernelFailure("GradOuter", err)
		} else {
			return
		}
	}
	refGradOuter(dl.in, dl.out, dl.delta, dl.input, dst, scale)
}

// refMatVec is the internal reference forward product; see cpu.Backend.MatVec.
func refMatVec[T utils.Float](w []T, in, out int, input, preact []T) {
	for o := range out {
		row := w[o*in : (o+1)*in : (o+1)*in]
		var sum T
		for j, v := range row {
			sum += v * input[j]
		}
		preact[o] = sum
	}
}

// refMatVecT is the internal reference transposed product; see cpu.Backend.MatVecT.
func refMatVecT[T utils.Float](w []T, in, out int, delta, dInput []T) {
	clear(dInput)
	for o := range out {
		d := delta[o]
		if d == 0 {
			continue
		}
		row := w[o*in : (o+1)*in : (o+1)*in]
		for j, v := range row {
			dInput[j] += d * v
		}
	}
}

// refGradOuter is the internal reference outer product; see cpu.Backend.GradOuter.
func refGradOuter[T utils.Float](in, out int, delta, input, dst []T, scale T) {
	for o := range out {
		coef := scale * delta[o]
		row := dst[o*in : (o+1)*in : (o+1)*in]
		for j := range row {
			row[j] = coef * input[j]
		}
	}
}

// CalculateValues runs the forward pass left-to-right over the structure-of-
// arrays store: each dense layer's pre-activations come from one matrix-vector
// product, the activation dispatcher is applied element-wise, and the
// normalization and dropout hooks gate the values the NEXT layer consumes.
//
// Post-activation values and output residuals are mirrored back into the cell
// graph so introspection APIs (HiddenActivations, visualization snapshots,
// tests) keep working. The mirror is strictly write-only: the math never reads
// a cell back, so the store stays the single source of truth.
//
// AI-Meta:
//   - Purpose: Execute the full forward pass, writing activations into the store and mirroring them into cells.
//   - Concurrency: NotSafe; mutates shared layer buffers. Use InferDense for concurrent reads.
//   - Related: [CalculateMisses], [CalculateWeights], [InferDense], [Train].
func (n *Network[T]) CalculateValues() {
	if !n.storeReady() {
		return
	}
	for i := range n.Hiddens {
		dl := &n.dense[i]
		if i == 0 {
			for j, c := range n.Input.cells {
				dl.input[j] = *c.GetValue()
			}
		} else {
			dl.loadInput(n.dense[i-1].act)
		}
		n.matVec(dl)

		act := n.hiddenActs[i]
		for o := range dl.out {
			dl.act[o] = activation.Activation(dl.preact[o], act)
		}
		// Per-layer normalization: the next layer consumes the NORMALIZED
		// values. The pre-activation buffer keeps the un-normalized linear sum —
		// σ′ in the backward pass applies to the activation, not the norm output.
		if nl, ok := n.normLayers[i]; ok {
			copy(dl.act, nl.Forward(dl.act))
		}
		// Honest dropout: the mask gates the values the NEXT layer consumes
		// (audit B3: the old flat mask ran after the whole forward pass and
		// never influenced the output). Training passes only — inference keeps
		// every unit (REG-3).
		if n.training && n.masker != nil {
			copy(dl.act, n.masker.MaskForwardLayer(i, dl.act))
		}
		for o, h := range n.Hiddens[i].cells {
			h.SetValue(dl.act[o])
		}
	}

	dl := &n.dense[len(n.Hiddens)]
	dl.loadInput(n.dense[len(n.Hiddens)-1].act)
	n.matVec(dl) // dl.preact aliases n.preactOutput

	// Output activation: true vector softmax, else element-wise dispatch.
	if n.outputAct == activation.SOFTMAX {
		activation.SoftmaxInto(n.outActBuf, n.preactOutput)
		copy(dl.act, n.outActBuf)
	} else {
		for o := range dl.out {
			dl.act[o] = activation.Activation(dl.preact[o], n.outputAct)
		}
	}
	// Residual miss = −∂ℓ/∂y so backprop drives the CONFIGURED loss (not MSE
	// for everyone). Fused pairs use the (t − y) shortcut whose σ′ is folded
	// analytically; outputEffDeriv returns 1 for them.
	for o, c := range n.Output.cells {
		c.SetValue(dl.act[o])
		t := c.GetTarget()
		if t == nil {
			// No label for this sample: no error signal. Explicitly zeroing
			// beats leaving the previous sample's residual in a reused buffer.
			dl.miss[o] = 0
			c.SetMiss(0)
			continue
		}
		if n.fusedOutput {
			dl.miss[o] = *t - dl.act[o]
		} else {
			dl.miss[o] = -loss.Derivative(dl.act[o], *t, n.lossMode)
		}
		c.SetMiss(dl.miss[o])
	}
}

// CalculateLoss computes the aggregate loss across Output cells using the
// supplied loss mode. Reads the activations written by CalculateValues.
//
// AI-Meta:
//   - Purpose: Compute scalar training loss after a forward pass; useful for logging or early stopping.
//   - Concurrency: ReadSafe.
//   - Related: [CalculateLossDefault], [CalculateValues], [loss.Aggregate].
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

// CalculateMisses runs the backward pass right-to-left across the dense chain.
// Output residuals are already set by CalculateValues.
//
// The recurrence per layer is:
//
//	δ_L    = σ′(preact_L) ⊙ miss_L            (output: effDeriv ⊙ miss)
//	miss_i = (W_{i+1}ᵀ · δ_{i+1})[:cells_i]   — the bias column has no upstream cell
//	miss_i = normBackward(maskBackward(miss_i))
//
// Folding the SOURCE layer's activation derivative into δ before propagating is
// the chain rule; omitting it (the historical bug) dropped one σ′ factor per
// layer crossed and corrupted every multi-layer gradient.
//
// AI-Meta:
//   - Purpose: Propagate error signals backward through all dense layers.
//   - Concurrency: NotSafe; mutates shared layer buffers and mirrors miss into cells.
//   - Related: [CalculateValues], [CalculateWeights], [AppendFlatGradients].
func (n *Network[T]) CalculateMisses() {
	if !n.storeReady() {
		return
	}
	out := &n.dense[len(n.Hiddens)]
	for o := range out.out {
		out.delta[o] = out.miss[o] * n.outputEffDeriv(o)
	}

	for i := len(n.Hiddens) - 1; i >= 0; i-- {
		src := &n.dense[i+1]
		dl := &n.dense[i]
		n.matVecT(src)
		copy(dl.miss, src.dInput[:src.sourceCount()])

		// Reverse of the forward's post-processing chain (norm → mask): first
		// undo the dropout gate — dropped units get zero miss, retained units
		// scale by 1/p. Elementwise scaling is sign-agnostic, so the miss
		// convention passes through unchanged.
		if n.training && n.masker != nil {
			copy(dl.miss, n.masker.MaskBackwardLayer(i, dl.miss))
		}
		// Route the accumulated miss through the layer's normalizer: what
		// arrived is −∂L/∂(norm output); downstream consumers need
		// −∂L/∂(activation). FeatureNorm.Backward works in the true-gradient
		// convention (it also accumulates ∂L/∂γ, ∂L/∂β), so the miss is negated
		// on the way in and the result negated on the way out.
		if nl, ok := n.normLayers[i]; ok {
			for j := range dl.miss {
				dl.miss[j] = -dl.miss[j]
			}
			dx := nl.Backward(dl.miss)
			for j := range dl.miss {
				dl.miss[j] = -dx[j]
			}
		}

		act := n.hiddenActs[i]
		for o := range dl.out {
			dl.delta[o] = dl.miss[o] * activation.Derivative(dl.preact[o], act)
		}
		for o, h := range n.Hiddens[i].cells {
			h.SetMiss(dl.miss[o])
		}
	}
}

// CalculateWeights applies one inline gradient-descent step to every dense
// layer: W[o][j] += rate · δ_o · input_j. Used by the bare Network.Train path;
// the pkg/nn facade instead exports flat gradients so a pluggable optimizer,
// scheduler, and regularizer can drive the update.
//
// AI-Meta:
//   - Purpose: Apply a vanilla SGD update to all layer weights from the current deltas.
//   - Concurrency: NotSafe; mutates the weight store in place.
//   - Related: [CalculateMisses], [AppendFlatGradients], [Train].
func (n *Network[T]) CalculateWeights(rate *T) {
	if !n.storeReady() {
		return
	}
	for li := range n.dense {
		dl := &n.dense[li]
		for o := range dl.out {
			coef := *rate * dl.delta[o]
			row := dl.store.W[o*dl.in : (o+1)*dl.in : (o+1)*dl.in]
			for j := range row {
				row[j] += coef * dl.input[j]
			}
		}
	}
}

// InferDense runs a stateless forward pass over the dense graph and returns a
// freshly allocated output slice. Unlike CalculateValues it touches NO shared
// state — not cell values, not the layers' scratch buffers — so any number of
// goroutines may call it concurrently under a read lock (audit D1: the old
// Query mutated shared cells and returned wrong answers under load). Weights
// are only read; the caller guarantees no concurrent Train via the facade's
// RWMutex.
//
// AI-Meta:
//   - Purpose: Read-only forward pass for concurrent inference; allocates its own scratch.
//   - Errors: ErrInputData (length mismatch or non-finite input).
//   - Concurrency: ReadSafe.
//   - Related: [CalculateValues], [nn.NN.Query].
func (n *Network[T]) InferDense(input []T) ([]T, error) {
	if len(input) != n.Input.Len() {
		return nil, utils.Newf(utils.ErrInputData,
			"InferDense: expected %d values, got %d", n.Input.Len(), len(input))
	}
	if !n.storeReady() {
		return nil, utils.Newf(utils.ErrUserConfig,
			"InferDense: network is not built — call Build before inference")
	}
	for j, v := range input {
		if !isFinite(v) {
			return nil, utils.Newf(utils.ErrInputData,
				"InferDense: non-finite input at index %d", j)
		}
	}
	cur := input
	for i := range n.Hiddens {
		dl := &n.dense[i]
		in := make([]T, dl.in)
		copy(in, cur[:dl.sourceCount()])
		if dl.hasBias {
			in[dl.in-1] = 1
		}
		preact := make([]T, dl.out)
		refMatVec(dl.store.W, dl.in, dl.out, in, preact)
		act := make([]T, dl.out)
		for o := range dl.out {
			act[o] = activation.Activation(preact[o], n.hiddenActs[i])
		}
		// Normalization in the read-only path uses ForwardInference — pure by
		// contract (no running-stat updates, no caches) so concurrent Query
		// goroutines can share the normalizer instance.
		if nl, ok := n.normLayers[i]; ok {
			copy(act, nl.ForwardInference(act))
		}
		cur = act
	}
	dl := &n.dense[len(n.Hiddens)]
	in := make([]T, dl.in)
	copy(in, cur[:dl.sourceCount()])
	if dl.hasBias {
		in[dl.in-1] = 1
	}
	preout := make([]T, dl.out)
	refMatVec(dl.store.W, dl.in, dl.out, in, preout)

	out := make([]T, dl.out)
	if n.outputAct == activation.SOFTMAX {
		activation.SoftmaxInto(out, preout)
	} else {
		for o := range preout {
			out[o] = activation.Activation(preout[o], n.outputAct)
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
// weight and writes it into dst (reusing the backing array when capacity
// allows). Gradients are emitted in the canonical order shared with
// AppendFlatWeights and with the weight store itself: Hiddens[0] → Hiddens[n-1]
// → Output, row-major within each layer.
//
// Definition: grad[o][j] = −δ_o × input_j. The sign convention ensures that
// SGD.Step (w -= lr × grad) reproduces the inline update in CalculateWeights
// (w += lr × δ × input).
//
// Must be called after CalculateMisses — it reads the deltas set there.
//
// AI-Meta:
//   - Purpose: Compute flat raw gradients for the optimizer Step call.
//   - Concurrency: NotSafe; reads shared layer buffers.
//   - Related: [AppendFlatWeights], [ApplyFlatWeights], [CalculateMisses].
//   - Stability: Stable.
func (n *Network[T]) AppendFlatGradients(dst []T) []T {
	if !n.storeReady() {
		return dst[:0]
	}
	total := len(n.weights)
	if cap(dst) >= total {
		dst = dst[:total]
	} else {
		dst = make([]T, total)
	}
	off := 0
	for li := range n.dense {
		dl := &n.dense[li]
		size := dl.in * dl.out
		n.gradOuter(dl, dst[off:off+size], -1)
		off += size
	}
	return dst
}

// AppendInputGradient computes ∂L/∂(input cell value) for every Input cell and
// writes the values into dst (reusing its backing array when capacity allows).
// Must be called AFTER [CalculateMisses] — it reads the first hidden layer's
// delta.
//
// For input cell i the formula is:
//
//	∂L/∂x_i = − Σ_{j in Hiddens[0]} δ_j · W₀[j][i]
//
// The sign convention matches [AppendFlatGradients]: SGD applies w -= lr · grad,
// so the negation here lets an upstream stage (e.g. a conv prefix) drive its own
// Backward pass with a consistent gradient.
//
// AI-Meta:
//   - Purpose: Expose the input-cell gradient so an upstream stage (conv prefix) can run its backward pass.
//   - Usage: gradX := n.AppendInputGradient(gradX[:0]).
//   - Concurrency: NotSafe; must run after CalculateMisses.
//   - Related: [CalculateMisses], [AppendFlatGradients], [SetInputs].
//   - Stability: Stable.
func (n *Network[T]) AppendInputGradient(dst []T) []T {
	inLen := n.Input.Len()
	if inLen == 0 || len(n.Hiddens) == 0 || !n.storeReady() {
		return dst[:0]
	}
	dl := &n.dense[0]
	n.matVecT(dl)
	if cap(dst) >= inLen {
		dst = dst[:inLen]
	} else {
		dst = make([]T, inLen)
	}
	for i := range inLen {
		dst[i] = -dl.dInput[i]
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
	lossValue := n.CalculateLossDefault()
	n.CalculateMisses()
	rate := n.LearningRate
	n.CalculateWeights(&rate)
	return lossValue, nil
}
