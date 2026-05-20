package transformer

import (
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/layer/attention"
	"github.com/teratron/gonn/pkg/utils"
)

// Block is the package-private interface satisfied by both [EncoderBlock] and
// [DecoderBlock]. [Stack] holds its children uniformly via this interface so
// encoder and decoder blocks may be composed without a type switch.
//
// Block extends [layer.Layer] with padding-mask forwarding (inherited from
// [attention.MaskedLayer]) and the inline-SGD weight-update hook [ApplyGradSGD].
type Block[T utils.Float] interface {
	layer.Layer[T]
	attention.MaskedLayer[T]
	// ApplyGradSGD fans the inline SGD step (w -= lr·g) to every child
	// parameter group and zeros gradient buffers.
	ApplyGradSGD(lr T)
}

// addInPlace adds src into dst element-wise (TRANS-6 residual connection).
// No scaling, clipping, or normalization — pure element-wise sum per invariant.
func addInPlace[T utils.Float](dst, src []T) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// ffn is an unexported position-wise feed-forward network applied to every
// position of a flat [seqLen × dmodel] tensor with SHARED weights:
//
//	y[p] = W2 · act(W1·x[p] + b1) + b2    for p ∈ [0, seqLen)
//
// Weight layout (row-major flat):
//
//	W1: [Dff × Dmodel]  — W1[i*Dmodel+j] is weight from input j to hidden i
//	W2: [Dmodel × Dff]  — W2[i*Dff+j]   is weight from hidden j to output i
//
// Per-position caches [seqLen×dmodel] and [seqLen×dff] are pre-allocated in
// Init and reused across Forward calls for zero-allocation hot path.
// Backward accumulates gradients across all positions into the shared weight
// gradient buffers, which is correct for parameter-sharing.
type ffn[T utils.Float] struct {
	W1  []T // [Dff * Dmodel]
	b1  []T // [Dff]
	W2  []T // [Dmodel * Dff]
	b2  []T // [Dmodel]
	gW1 []T
	gb1 []T
	gW2 []T
	gb2 []T
	// per-position caches (seqLen × dmodel and seqLen × dff)
	cacheX  []T // [seqLen * dmodel]
	cacheZ1 []T // [seqLen * dff]
	cacheH  []T // [seqLen * dff]
	seqLen  int
	dmodel  int
	dff     int
	actMode activation.Type
}

// newFFN allocates an ffn with zeroed weights. Call Init(rng) before use to
// Xavier-initialize the weight matrices and pre-allocate per-position caches.
// actMode zero-value (ELISH) is replaced by ReLU by the caller
// (TransformerConfig constructor) per TRANS-C4.
func newFFN[T utils.Float](seqLen, dmodel, dff int, actMode activation.Type) *ffn[T] {
	return &ffn[T]{
		W1:      make([]T, dff*dmodel),
		b1:      make([]T, dff),
		W2:      make([]T, dmodel*dff),
		b2:      make([]T, dmodel),
		gW1:     make([]T, dff*dmodel),
		gb1:     make([]T, dff),
		gW2:     make([]T, dmodel*dff),
		gb2:     make([]T, dmodel),
		seqLen:  seqLen,
		dmodel:  dmodel,
		dff:     dff,
		actMode: actMode,
	}
}

// Init Xavier-initializes the W1 and W2 weight matrices and pre-allocates
// per-position caches. Called from EncoderBlock.Init / DecoderBlock.Init
// with a forked RNG stream.
func (f *ffn[T]) Init(rng *rand.Rand) {
	for i := range f.W1 {
		f.W1[i] = utils.XavierUniform[T](rng, f.dmodel, f.dff)
	}
	for i := range f.W2 {
		f.W2[i] = utils.XavierUniform[T](rng, f.dff, f.dmodel)
	}
	f.cacheX = make([]T, f.seqLen*f.dmodel)
	f.cacheZ1 = make([]T, f.seqLen*f.dff)
	f.cacheH = make([]T, f.seqLen*f.dff)
}

// Forward computes y[p] = W2·act(W1·x[p]+b1)+b2 for every position p ∈ [0,seqLen)
// and caches all per-position intermediates for Backward.
// x has shape [seqLen*Dmodel]; returned y has the same shape.
func (f *ffn[T]) Forward(x []T) []T {
	// Lazy cache allocation supports cloneFFN (which skips Init).
	if f.cacheX == nil {
		f.cacheX = make([]T, f.seqLen*f.dmodel)
		f.cacheZ1 = make([]T, f.seqLen*f.dff)
		f.cacheH = make([]T, f.seqLen*f.dff)
	}
	copy(f.cacheX, x)
	y := make([]T, f.seqLen*f.dmodel)
	for p := 0; p < f.seqLen; p++ {
		xp := x[p*f.dmodel : (p+1)*f.dmodel]
		z1p := f.cacheZ1[p*f.dff : (p+1)*f.dff]
		hp := f.cacheH[p*f.dff : (p+1)*f.dff]
		yp := y[p*f.dmodel : (p+1)*f.dmodel]
		// first linear: z1p = W1·xp + b1
		for i := range z1p {
			z1p[i] = f.b1[i]
			base := i * f.dmodel
			for j, xj := range xp {
				z1p[i] += f.W1[base+j] * xj
			}
		}
		// activation: hp = act(z1p)
		copy(hp, z1p)
		applyActivationInPlace(f.actMode, hp)
		// second linear: yp = W2·hp + b2
		for i := range yp {
			yp[i] = f.b2[i]
			base := i * f.dff
			for j, hj := range hp {
				yp[i] += f.W2[base+j] * hj
			}
		}
	}
	return y
}

// Backward accumulates parameter gradients across all seqLen positions and
// returns ∂L/∂x with shape [seqLen*Dmodel]. Must be called after Forward.
func (f *ffn[T]) Backward(upstream []T) []T {
	dx := make([]T, f.seqLen*f.dmodel)
	for p := 0; p < f.seqLen; p++ {
		up := upstream[p*f.dmodel : (p+1)*f.dmodel]
		xp := f.cacheX[p*f.dmodel : (p+1)*f.dmodel]
		z1p := f.cacheZ1[p*f.dff : (p+1)*f.dff]
		hp := f.cacheH[p*f.dff : (p+1)*f.dff]
		dxp := dx[p*f.dmodel : (p+1)*f.dmodel]
		// second linear backward
		dh := make([]T, f.dff)
		for i, u := range up {
			f.gb2[i] += u
			base := i * f.dff
			for j, hj := range hp {
				f.gW2[base+j] += u * hj
				dh[j] += u * f.W2[base+j]
			}
		}
		// activation backward — pass pre-activation z1p per ReLU convention
		dz1 := make([]T, f.dff)
		for j := range dz1 {
			dz1[j] = dh[j] * activation.Derivative(z1p[j], f.actMode)
		}
		// first linear backward
		for i, d := range dz1 {
			f.gb1[i] += d
			base := i * f.dmodel
			for j, xj := range xp {
				f.gW1[base+j] += d * xj
				dxp[j] += d * f.W1[base+j]
			}
		}
	}
	return dx
}

// ApplyGradSGD performs w -= lr·g elementwise on all four parameter tensors
// and zeros the gradient buffers after update.
func (f *ffn[T]) ApplyGradSGD(lr T) {
	for i := range f.W1 {
		f.W1[i] -= lr * f.gW1[i]
		f.gW1[i] = 0
	}
	for i := range f.b1 {
		f.b1[i] -= lr * f.gb1[i]
		f.gb1[i] = 0
	}
	for i := range f.W2 {
		f.W2[i] -= lr * f.gW2[i]
		f.gW2[i] = 0
	}
	for i := range f.b2 {
		f.b2[i] -= lr * f.gb2[i]
		f.gb2[i] = 0
	}
}

// GradSlots returns the gradient buffers for W1+W2 (weights) and b1+b2 (biases)
// as a convenience for the EncoderBlock gradient aggregation.
// Caller must not modify the returned slices directly.
func (f *ffn[T]) GradSlots() (gradW, gradB []T) {
	// Concatenate into caller-owned views using append; avoids an extra alloc
	// per call by letting the caller manage the destination buffer.
	gradW = append(f.gW1, f.gW2...)
	gradB = append(f.gb1, f.gb2...)
	return gradW, gradB
}
