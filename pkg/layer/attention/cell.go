package attention

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// softmaxRowwise applies numerically stable softmax over each row of an
// [n, k] matrix stored as flat row-major []T. Mutates scores in place.
// Per-row max-subtract before exp prevents overflow (ATT-3 numerical stability).
func softmaxRowwise[T utils.Float](scores []T, n, k int) {
	for i := range n {
		base := i * k
		maxVal := scores[base]
		for j := 1; j < k; j++ {
			if scores[base+j] > maxVal {
				maxVal = scores[base+j]
			}
		}
		var sum T
		for j := range k {
			v := T(math.Exp(float64(scores[base+j] - maxVal)))
			scores[base+j] = v
			sum += v
		}
		if sum == 0 {
			continue
		}
		for j := range k {
			scores[base+j] /= sum
		}
	}
}

// softmaxRowwiseWithMask applies softmax row-wise, treating positions where
// mask[j] = false as -Inf (zero softmax weight). When mask == nil it behaves
// identically to softmaxRowwise.
//
// Implementation: set masked positions to -Inf, then run softmaxRowwise.
// In IEEE 754, exp(-Inf - rowMax) = 0 exactly, so masked positions produce
// zero weight without a separate branch.
func softmaxRowwiseWithMask[T utils.Float](scores []T, n, k int, mask []bool) {
	if mask == nil {
		softmaxRowwise(scores, n, k)
		return
	}
	negInf := T(math.Inf(-1))
	for i := range n {
		base := i * k
		for j := range k {
			if !mask[j] {
				scores[base+j] = negInf
			}
		}
	}
	softmaxRowwise(scores, n, k)
}

// softmaxBackwardRowwise computes the Jacobian-vector product for row-wise softmax:
//
//	dScores[i, j] = (dA[i, j] - <dA[i, :], A[i, :]>) * A[i, j]
//
// Writes into dScores in place. dA and A must both be [n, k] flat row-major.
func softmaxBackwardRowwise[T utils.Float](dA, A, dScores []T, n, k int) {
	for i := range n {
		base := i * k
		var dot T
		for j := range k {
			dot += dA[base+j] * A[base+j]
		}
		for j := range k {
			dScores[base+j] = (dA[base+j] - dot) * A[base+j]
		}
	}
}

// softmaxBackwardRowwiseWithMask handles masked positions: since A[i,j] = 0
// for any masked j (set by softmaxRowwiseWithMask), the Jacobian-vector product
// automatically produces dScores[i,j] = 0 for those positions without additional
// logic. This function delegates to softmaxBackwardRowwise.
func softmaxBackwardRowwiseWithMask[T utils.Float](dA, A, dScores []T, n, k int, _ []bool) {
	softmaxBackwardRowwise(dA, A, dScores, n, k)
}

// attentionScores fills scores[h, i, j] = (Q[h, i, :] · K[h, j, :]) * scale.
// All buffers use head-major flat layout: [numHeads, seqLen, dk] index as
// h*seqLen*dk + t*dk + k; scores is [numHeads, seqLen, seqLen] as
// h*seqLen*seqLen + i*seqLen + j.
func attentionScores[T utils.Float](Q, K, scores []T, numHeads, seqLen, dk int, scale T) {
	for h := range numHeads {
		hQ := h * seqLen * dk
		hS := h * seqLen * seqLen
		for i := range seqLen {
			for j := range seqLen {
				var dot T
				qi := hQ + i*dk
				kj := hQ + j*dk
				for d := range dk {
					dot += Q[qi+d] * K[kj+d]
				}
				scores[hS+i*seqLen+j] = dot * scale
			}
		}
	}
}

// applyCausalMask sets scores[h, i, j] = -Inf for all j > i across all heads.
// scores is [numHeads, seqLen, seqLen] head-major flat. Prevents each query
// position from attending to future key positions (ATT-5).
func applyCausalMask[T utils.Float](scores []T, numHeads, seqLen int) {
	negInf := T(math.Inf(-1))
	for h := range numHeads {
		off := h * seqLen * seqLen
		for i := range seqLen {
			for j := i + 1; j < seqLen; j++ {
				scores[off+i*seqLen+j] = negInf
			}
		}
	}
}

// applyPaddingMask sets scores[h, i, j] = -Inf for all j where mask[j] = false,
// broadcasting over the query axis (ATT-6). Both causal and padding masks compose
// additively: callers apply causal first, then padding.
func applyPaddingMask[T utils.Float](scores []T, mask []bool, numHeads, seqLen int) {
	negInf := T(math.Inf(-1))
	for h := range numHeads {
		off := h * seqLen * seqLen
		for i := range seqLen {
			for j := range seqLen {
				if !mask[j] {
					scores[off+i*seqLen+j] = negInf
				}
			}
		}
	}
}

// projMat computes out[t, d] = bias[d] + sum_i src[t, i] * W[d, i] for all
// t in [0, seqLen) and d in [0, dmodel). Implements a linear projection
// (X @ W^T + b) for the Q/K/V/O weight matrices in MultiHeadAttention.
// out must be pre-allocated to seqLen*dmodel.
func projMat[T utils.Float](src, W, bias, out []T, seqLen, dmodel int) {
	for t := range seqLen {
		for d := range dmodel {
			v := bias[d]
			base := t * dmodel
			row := d * dmodel
			for i := range dmodel {
				v += src[base+i] * W[row+i]
			}
			out[t*dmodel+d] = v
		}
	}
}

// contextMul computes ctx[h, t, d] = sum_j A[h, t, j] * V[h, j, d] for all
// heads, query positions, and head dimensions. All buffers are head-major flat.
func contextMul[T utils.Float](A, V, ctx []T, numHeads, seqLen, dk int) {
	for h := range numHeads {
		ha := h * seqLen * seqLen
		hv := h * seqLen * dk
		hc := h * seqLen * dk
		for t := range seqLen {
			for d := range dk {
				var acc T
				for j := range seqLen {
					acc += A[ha+t*seqLen+j] * V[hv+j*dk+d]
				}
				ctx[hc+t*dk+d] = acc
			}
		}
	}
}

// splitHeads reindexes src from [seqLen, numHeads*dk] (sequence-major) into
// dst with [numHeads, seqLen, dk] (head-major) layout. No allocation — dst
// must be pre-allocated with length numHeads*seqLen*dk.
//
// src[t, h*dk + d]  →  dst[h, t, d]
// src index: t*(numHeads*dk) + h*dk + d
// dst index: h*seqLen*dk + t*dk + d
func splitHeads[T utils.Float](src, dst []T, numHeads, seqLen, dk int) {
	dmodel := numHeads * dk
	for t := range seqLen {
		for h := range numHeads {
			for d := range dk {
				dst[h*seqLen*dk+t*dk+d] = src[t*dmodel+h*dk+d]
			}
		}
	}
}

// joinHeads is the inverse of splitHeads: reindexes dst from head-major
// [numHeads, seqLen, dk] back to sequence-major [seqLen, numHeads*dk].
//
// src[h, t, d]  →  dst[t, h*dk + d]
func joinHeads[T utils.Float](src, dst []T, numHeads, seqLen, dk int) {
	dmodel := numHeads * dk
	for h := range numHeads {
		for t := range seqLen {
			for d := range dk {
				dst[t*dmodel+h*dk+d] = src[h*seqLen*dk+t*dk+d]
			}
		}
	}
}
