package activation

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// SoftmaxInto writes the numerically-stable softmax of preact into dst:
// dst[i] = exp(preact[i] − max) / Σ exp(preact[j] − max). dst and preact must
// have equal length; dst may alias preact. Used by the output layer when the
// output activation is SOFTMAX so the head produces a true probability
// distribution (Σ dst = 1) rather than the element-wise sigmoid stand-in.
//
// AI-Meta:
//   - Purpose: Stable vector softmax over a full output layer.
//   - Usage: activation.SoftmaxInto(out, preact).
//   - Related: [Activation], [Type].
//   - Stability: Stable.
func SoftmaxInto[T utils.Float](dst, preact []T) {
	if len(preact) == 0 {
		return
	}
	maxVal := preact[0]
	for _, v := range preact[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum T
	for i, v := range preact {
		e := T(math.Exp(float64(v - maxVal)))
		dst[i] = e
		sum += e
	}
	if sum == 0 {
		sum = 1
	}
	for i := range dst {
		dst[i] /= sum
	}
}
