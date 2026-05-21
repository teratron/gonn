package quantization

import (
	"fmt"
	"math"

	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// EvalSample pairs an input with its expected output for side-by-side evaluation.
//
// AI-Meta:
//   - Purpose: One (input, expected) pair for Evaluate[T] (T-19A09).
//   - Stability: Experimental.
type EvalSample[T utils.Float] struct {
	Input    []T
	Expected []T
}

// EvaluationResult reports the side-by-side accuracy comparison between a
// float baseline network and its quantized counterpart.
//
// AI-Meta:
//   - Purpose: Output of Evaluate[T] — informational only, no auto-accept gate (QUANT-C6).
//   - Stability: Experimental.
type EvaluationResult struct {
	PerLayerL2    []float64
	MetricFloat   float64
	MetricQuant   float64
	DeltaRelative float64
}

// Evaluate runs baseline and quantized networks on every sample in evalSet,
// computes metricFn for both, and returns an EvaluationResult.
// Returns an error for an empty evalSet.
//
// AI-Meta:
//   - Purpose: Side-by-side float vs int8 accuracy gate (QUANT-8 / T-19A09).
//   - Errors: ErrInputData for empty evalSet.
//   - Stability: Experimental.
func Evaluate[T utils.Float](
	baseline *nn.NN[T],
	quantized *QuantizedNetwork[T],
	evalSet []EvalSample[T],
	metricFn func(output, expected []T) float64,
) (EvaluationResult, error) {
	if len(evalSet) == 0 {
		return EvaluationResult{}, fmt.Errorf("evaluate: empty eval set: %w", utils.ErrInputData)
	}

	nLayers := len(quantized.Layers)
	perLayerSumSq := make([]float64, nLayers)
	perLayerNormSq := make([]float64, nLayers)

	var metricSumFloat, metricSumQuant float64

	prefix := baseline.Config().ConvPrefix

	for _, sample := range evalSet {
		// Baseline forward.
		floatOut, err := baseline.Query(sample.Input)
		if err != nil {
			return EvaluationResult{}, fmt.Errorf("evaluate: baseline query: %w", err)
		}
		metricSumFloat += metricFn(floatOut, sample.Expected)

		// Quantized forward.
		quantOut := quantized.Forward(sample.Input)
		metricSumQuant += metricFn(quantOut, sample.Expected)

		// Per-layer L2 delta: drive prefix manually, compare at each layer output.
		if len(prefix) == nLayers {
			cur := sample.Input
			for i, layer := range prefix {
				floatLayerOut := layer.Forward(cur)
				quantLayerOut := quantized.Layers[i].Forward(cur)

				var diffSq, normSq float64
				for j := range floatLayerOut {
					var qv float64
					if j < len(quantLayerOut) {
						qv = float64(quantLayerOut[j])
					}
					d := float64(floatLayerOut[j]) - qv
					diffSq += d * d
					normSq += float64(floatLayerOut[j]) * float64(floatLayerOut[j])
				}
				perLayerSumSq[i] += diffSq
				perLayerNormSq[i] += normSq

				cur = floatLayerOut // advance with float output for fair comparison
			}
		}
	}

	n := float64(len(evalSet))
	metricFloat := metricSumFloat / n
	metricQuant := metricSumQuant / n

	var deltaRel float64
	if math.Abs(metricFloat) > 1e-10 {
		deltaRel = (metricQuant - metricFloat) / math.Abs(metricFloat)
	}

	perLayerL2 := make([]float64, nLayers)
	for i := range nLayers {
		if perLayerNormSq[i] > 1e-10 {
			perLayerL2[i] = math.Sqrt(perLayerSumSq[i] / perLayerNormSq[i])
		}
	}

	return EvaluationResult{
		MetricFloat:   metricFloat,
		MetricQuant:   metricQuant,
		DeltaRelative: deltaRel,
		PerLayerL2:    perLayerL2,
	}, nil
}
