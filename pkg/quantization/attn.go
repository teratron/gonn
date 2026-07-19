package quantization

import (
	"math"

	attentionpkg "github.com/teratron/gonn/pkg/layer/attention"
	"github.com/teratron/gonn/pkg/utils"
)

// QuantizedAttentionProjections is a weight-only int8 quantized multi-head
// attention layer. Only the four projection matrices (Wq/Wk/Wv/Wo) are
// quantized; scoring and softmax remain float (QUANT-10).
//
// AI-Meta:
//   - Purpose: Weight-only int8 attention projection layer for Phase γ (T-19A10).
//   - Stability: Experimental.
type QuantizedAttentionProjections[T utils.Float] struct {
	Bq       []float64
	Bk       []float64
	Bo       []float64
	Bv       []float64
	Wq       []int8
	Wk       []int8
	Wo       []int8
	Wv       []int8
	WqParams QuantizationParams
	WkParams QuantizationParams
	WoParams QuantizationParams
	WvParams QuantizationParams
	Dmodel   int
	NumHeads int
	SeqLen   int
	Causal   bool
}

// Forward applies multi-head attention with on-the-fly weight dequantization.
// Input must be a sequence-major flat slice of length SeqLen×Dmodel.
// Returns a flat slice of the same shape.
func (q *QuantizedAttentionProjections[T]) Forward(x []T) []T {
	if len(x) != q.SeqLen*q.Dmodel {
		return []T{}
	}
	dk := q.Dmodel / q.NumHeads
	scale := T(1.0 / math.Sqrt(float64(dk)))

	projQ := attnDequantProject(x, q.Wq, q.WqParams, q.Bq, q.SeqLen, q.Dmodel)
	projK := attnDequantProject(x, q.Wk, q.WkParams, q.Bk, q.SeqLen, q.Dmodel)
	projV := attnDequantProject(x, q.Wv, q.WvParams, q.Bv, q.SeqLen, q.Dmodel)

	hQ := attnSplitHeads(projQ, q.NumHeads, q.SeqLen, dk)
	hK := attnSplitHeads(projK, q.NumHeads, q.SeqLen, dk)
	hV := attnSplitHeads(projV, q.NumHeads, q.SeqLen, dk)

	ss := q.SeqLen * q.SeqLen
	scores := make([]T, q.NumHeads*ss)
	for h := range q.NumHeads {
		qOff := h * q.SeqLen * dk
		kOff := h * q.SeqLen * dk
		sOff := h * ss
		for i := range q.SeqLen {
			for j := range q.SeqLen {
				var dot T
				for d := range dk {
					dot += hQ[qOff+i*dk+d] * hK[kOff+j*dk+d]
				}
				scores[sOff+i*q.SeqLen+j] = dot * scale
			}
		}
	}

	if q.Causal {
		for h := range q.NumHeads {
			off := h * ss
			for i := range q.SeqLen {
				for j := i + 1; j < q.SeqLen; j++ {
					scores[off+i*q.SeqLen+j] = T(math.Inf(-1))
				}
			}
		}
	}

	for h := range q.NumHeads {
		off := h * ss
		attnSoftmaxRows(scores[off:off+ss], q.SeqLen)
	}

	ctx := make([]T, q.NumHeads*q.SeqLen*dk)
	for h := range q.NumHeads {
		aOff := h * ss
		vOff := h * q.SeqLen * dk
		cOff := h * q.SeqLen * dk
		for t := range q.SeqLen {
			for d := range dk {
				var acc T
				for j := range q.SeqLen {
					acc += scores[aOff+t*q.SeqLen+j] * hV[vOff+j*dk+d]
				}
				ctx[cOff+t*dk+d] = acc
			}
		}
	}

	merged := attnJoinHeads(ctx, q.NumHeads, q.SeqLen, dk)
	return attnDequantProject(merged, q.Wo, q.WoParams, q.Bo, q.SeqLen, q.Dmodel)
}

// attnDequantProject computes out[t, d] = sum_k(dequantize(W[d,k]) * src[t,k]) + bias[d].
func attnDequantProject[T utils.Float](src []T, w []int8, params QuantizationParams, bias []float64, seqLen, dmodel int) []T {
	out := make([]T, seqLen*dmodel)
	for t := range seqLen {
		for d := range dmodel {
			var acc float64
			for k := range dmodel {
				acc += dequantize(w[d*dmodel+k], params, d) * float64(src[t*dmodel+k])
			}
			if len(bias) > d {
				acc += bias[d]
			}
			out[t*dmodel+d] = T(acc)
		}
	}
	return out
}

// attnSplitHeads reshapes sequence-major [SeqLen, Dmodel] to head-major [NumHeads, SeqLen, Dk].
func attnSplitHeads[T utils.Float](src []T, numHeads, seqLen, dk int) []T {
	dst := make([]T, numHeads*seqLen*dk)
	for t := range seqLen {
		for h := range numHeads {
			for d := range dk {
				dst[h*seqLen*dk+t*dk+d] = src[t*numHeads*dk+h*dk+d]
			}
		}
	}
	return dst
}

// attnJoinHeads is the inverse of attnSplitHeads.
func attnJoinHeads[T utils.Float](src []T, numHeads, seqLen, dk int) []T {
	dst := make([]T, seqLen*numHeads*dk)
	for t := range seqLen {
		for h := range numHeads {
			for d := range dk {
				dst[t*numHeads*dk+h*dk+d] = src[h*seqLen*dk+t*dk+d]
			}
		}
	}
	return dst
}

// attnSoftmaxRows applies softmax to each row of length n in a flat n×n slice.
func attnSoftmaxRows[T utils.Float](scores []T, n int) {
	for i := range n {
		row := scores[i*n : (i+1)*n]
		var maxV T
		for _, v := range row {
			if v > maxV {
				maxV = v
			}
		}
		var sum T
		for j, v := range row {
			row[j] = T(math.Exp(float64(v - maxV)))
			sum += row[j]
		}
		if sum > 0 {
			for j := range row {
				row[j] /= sum
			}
		}
	}
}

func newQuantizedAttentionProjections[T utils.Float](src *attentionpkg.MultiHeadAttention[T], cfg QuantizationConfig[T]) *QuantizedAttentionProjections[T] {
	dmodel := src.Dmodel
	cin := dmodel

	quantizeProj := func(w []T) ([]int8, QuantizationParams) {
		wf := toFloat64Slice(w)
		qw, params := quantizeWeightsTensor(wf, dmodel, cin, cfg.WeightGranularity)
		params.Strategy = cfg.Strategy
		return qw, params
	}

	qWq, pWq := quantizeProj(src.Wq)
	qWk, pWk := quantizeProj(src.Wk)
	qWv, pWv := quantizeProj(src.Wv)
	qWo, pWo := quantizeProj(src.Wo)

	return &QuantizedAttentionProjections[T]{
		WqParams: pWq,
		WkParams: pWk,
		WvParams: pWv,
		WoParams: pWo,
		Wq:       qWq,
		Wk:       qWk,
		Wv:       qWv,
		Wo:       qWo,
		Bq:       toFloat64Slice(src.Bq),
		Bk:       toFloat64Slice(src.Bk),
		Bv:       toFloat64Slice(src.Bv),
		Bo:       toFloat64Slice(src.Bo),
		Dmodel:   dmodel,
		NumHeads: src.NumHeads,
		SeqLen:   src.SeqLen,
		Causal:   src.Causal,
	}
}
