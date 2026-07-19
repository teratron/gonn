package loss

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// Derivative returns dℓ/d(predicted) for the element-wise loss selected by mode,
// where ℓ is the exact per-output loss returned by [Loss]. The training loop
// forms the output residual as miss = −Derivative(y, t, mode), so a correct
// derivative here is what makes each loss actually drive learning (previously
// only the residual (t − y) was used, i.e. every loss trained as MSE).
//
// Convention: `predicted` is the network output y (post output-activation);
// `target` is the label t. Vector losses (CCE, COSINE, CAT_HINGE) are handled
// via the fused softmax+cross-entropy shortcut in the training loop and via
// [VectorDerivative]; their element-wise entry here is a best-effort fallback.
//
// AI-Meta:
//   - Purpose: Element-wise loss gradient dℓ/dy used to build the output residual.
//   - Usage: miss := -loss.Derivative[float32](y, t, loss.MSE).
//   - Related: [Loss], [VectorDerivative], [Type].
//   - Stability: Stable.
func Derivative[T utils.Float](predicted, target T, mode Type) T {
	switch mode {
	case MSE:
		// d[½(y−t)²]/dy = (y − t). Matches the ½-scaled mseLoss.
		return predicted - target
	case MAE, AVG:
		return signT(predicted - target)
	case RMSE:
		// RMSE aggregates non-separably across outputs; train on the (½-)MSE
		// gradient and apply the sqrt only to the reported scalar.
		return predicted - target
	case BCE:
		p := clampUnit(predicted)
		return (p - target) / (p * (1 - p))
	case MSLE:
		p := predicted
		if p <= -1 {
			p = T(-1) + T(1e-7)
		}
		lp := T(math.Log(float64(p + 1)))
		lt := T(math.Log(float64(target + 1)))
		return 2 * (lp - lt) / (p + 1)
	case KLD:
		p := clampLow(predicted)
		if target <= 0 {
			return 0
		}
		return -target / p
	case POISSON:
		p := clampLow(predicted)
		return 1 - target/p
	case MAPE:
		denom := absT(target)
		if denom < T(1e-7) {
			denom = T(1e-7)
		}
		return signT(predicted-target) * 100 / denom
	case HINGE, CAT_HINGE:
		if 1-predicted*target > 0 {
			return -target
		}
		return 0
	case SQ_HINGE:
		m := 1 - predicted*target
		if m > 0 {
			return -2 * target * m
		}
		return 0
	case LOG_COSH:
		return T(math.Tanh(float64(predicted - target)))
	case HUBER:
		d := predicted - target
		if absT(d) <= 1 {
			return d
		}
		return signT(d)
	case ARCTAN:
		diff := float64(predicted - target)
		return T(1.0 / (1.0 + diff*diff))
	case CCE, CROSS_ENTROPY:
		// Exact only combined with the softmax Jacobian; the fused path in
		// the engine bypasses this. Element fallback: d(-t·log y)/dy.
		return -target / clampLow(predicted)
	case COSINE:
		return predicted - target // fallback; true cosine grad is vector-valued
	default:
		return 2 * (predicted - target)
	}
}

// VectorLoss computes a whole-vector loss for modes whose value cannot be
// decomposed per output. Used by the training-loss reporter and the gradient
// oracle for CCE / COSINE / CAT_HINGE. For element-wise modes it sums Loss over
// the vector so callers can use a single entry point.
//
// AI-Meta:
//   - Purpose: Whole-vector loss for cross-entropy / cosine families.
//   - Usage: l := loss.VectorLoss[float32](y, t, loss.CCE).
//   - Related: [Loss], [VectorDerivative].
//   - Stability: Stable.
func VectorLoss[T utils.Float](predicted, target []T, mode Type) T {
	switch mode {
	case CCE, CROSS_ENTROPY:
		var s T
		for i := range predicted {
			s += -target[i] * T(math.Log(float64(clampLow(predicted[i]))))
		}
		return s
	case COSINE:
		var dot, np, nt T
		for i := range predicted {
			dot += predicted[i] * target[i]
			np += predicted[i] * predicted[i]
			nt += target[i] * target[i]
		}
		den := T(math.Sqrt(float64(np))) * T(math.Sqrt(float64(nt)))
		if den < T(1e-12) {
			den = T(1e-12)
		}
		return 1 - dot/den // cosine distance
	case CAT_HINGE:
		// max over negative classes of (score - true_score + 1), hinged.
		var trueScore T
		for i := range target {
			if target[i] > 0 {
				trueScore = predicted[i]
			}
		}
		var s T
		for i := range predicted {
			if target[i] > 0 {
				continue
			}
			m := predicted[i] - trueScore + 1
			if m > 0 {
				s += m
			}
		}
		return s
	default:
		var s T
		for i := range predicted {
			s += Loss(predicted[i], target[i], mode)
		}
		return s
	}
}

// VectorDerivative returns dℓ/dy_i for whole-vector losses. For CCE with a
// softmax output the engine uses the fused (y − t) shortcut instead; this is
// provided for completeness and for non-fused callers.
//
// AI-Meta:
//   - Purpose: Whole-vector loss gradient for cross-entropy / cosine families.
//   - Related: [VectorLoss], [Derivative].
//   - Stability: Stable.
func VectorDerivative[T utils.Float](predicted, target []T, mode Type) []T {
	out := make([]T, len(predicted))
	switch mode {
	case CCE, CROSS_ENTROPY:
		for i := range predicted {
			out[i] = -target[i] / clampLow(predicted[i])
		}
	case CAT_HINGE:
		var trueScore T
		trueIdx := -1
		for i := range target {
			if target[i] > 0 {
				trueScore = predicted[i]
				trueIdx = i
			}
		}
		for i := range predicted {
			if target[i] > 0 {
				continue
			}
			if predicted[i]-trueScore+1 > 0 {
				out[i] += 1
				if trueIdx >= 0 {
					out[trueIdx] -= 1
				}
			}
		}
	default:
		for i := range predicted {
			out[i] = Derivative(predicted[i], target[i], mode)
		}
	}
	return out
}

func signT[T utils.Float](x T) T {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}

func absT[T utils.Float](x T) T {
	if x < 0 {
		return -x
	}
	return x
}

// clampLow floors a value at 1e-7 to keep logs and divisions finite.
func clampLow[T utils.Float](x T) T {
	if x < T(1e-7) {
		return T(1e-7)
	}
	return x
}

// clampUnit clamps a probability into [1e-7, 1-1e-7].
func clampUnit[T utils.Float](x T) T {
	eps := T(1e-7)
	if x < eps {
		return eps
	}
	if x > 1-eps {
		return 1 - eps
	}
	return x
}
