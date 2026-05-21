package quantization

import (
	"fmt"
	"math"
	"sort"

	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

const calibHardFloor = 32 // ErrCalibTooFewSamples below this

// layerStats accumulates per-layer activation statistics during calibration.
type layerStats struct {
	min, max float64
	vals     []float64 // raw values for percentile; cleared after Params()
}

// CalibrationRunner drives a source network's ConvPrefix layers sample-by-sample,
// collecting per-layer activation min/max statistics needed for full-int8 quantization.
//
// AI-Meta:
//   - Purpose: Phase β activation calibration — collects per-layer stats to compute ActParams (T-19A07).
//   - Stability: Experimental.
type CalibrationRunner[T utils.Float] struct {
	net   *nn.NN[T]
	cfg   QuantizationConfig[T]
	stats []layerStats // indexed by ConvPrefix position
}

// NewCalibrationRunner constructs a CalibrationRunner for the given network and config.
func NewCalibrationRunner[T utils.Float](net *nn.NN[T], cfg QuantizationConfig[T]) *CalibrationRunner[T] {
	n := len(net.Config().ConvPrefix)
	stats := make([]layerStats, n)
	for i := range stats {
		stats[i].min = math.MaxFloat64
		stats[i].max = -math.MaxFloat64
	}
	return &CalibrationRunner[T]{net: net, cfg: cfg, stats: stats}
}

// Run passes all samples through the ConvPrefix layer chain, recording per-layer
// activation statistics. Returns ErrCalibTooFewSamples when len(samples) < 32.
func (c *CalibrationRunner[T]) Run(samples [][]T) error {
	if len(samples) < calibHardFloor {
		return fmt.Errorf("calibration: %d samples < hard floor %d: %w",
			len(samples), calibHardFloor, utils.ErrCalibTooFewSamples)
	}

	prefix := c.net.Config().ConvPrefix
	for _, x := range samples {
		cur := x
		for i, layer := range prefix {
			// Record stats on the input to this layer (= output of previous).
			c.recordInput(i, cur)
			cur = layer.Forward(cur)
		}
	}
	return nil
}

// recordInput updates stats[i] with the values from an activation vector.
func (c *CalibrationRunner[T]) recordInput(i int, x []T) {
	s := &c.stats[i]
	for _, v := range x {
		f := float64(v)
		if f < s.min {
			s.min = f
		}
		if f > s.max {
			s.max = f
		}
		if c.cfg.Strategy == Percentile99p9 {
			s.vals = append(s.vals, f)
		}
	}
}

// Params computes QuantizationParams for each layer based on the collected stats
// and the configured CalibStrategy. Returns ErrDegenerateRange (wrapped) for any
// layer whose range is too small; the fallback scale=1.0, zp=0 is still returned.
func (c *CalibrationRunner[T]) Params() ([]QuantizationParams, error) {
	result := make([]QuantizationParams, len(c.stats))
	var firstDegen error

	for i, s := range c.stats {
		if s.min > s.max {
			// No data recorded for this layer.
			result[i] = degenerateParams()
			continue
		}
		rMin, rMax := s.min, s.max

		switch c.cfg.Strategy {
		case Percentile99p9:
			if len(s.vals) > 0 {
				rMin, rMax = percentileRange(s.vals, c.cfg.PercentileThreshold)
			}
		}

		if rMax-rMin < 1e-6 {
			if firstDegen == nil {
				firstDegen = fmt.Errorf("calibration layer %d: degenerate range [%.6g, %.6g]: %w",
					i, rMin, rMax, utils.ErrDegenerateRange)
			}
			result[i] = degenerateParams()
			continue
		}
		result[i] = computeSymmetric(rMin, rMax)
		result[i].Strategy = c.cfg.Strategy
	}
	return result, firstDegen
}

// percentileRange computes the (−p, +p) symmetric clipped range from a flat
// value slice, where p = threshold (e.g. 99.9 means clip at 99.9th percentile
// of |v|). Returns (−clip, +clip).
func percentileRange(vals []float64, threshold float64) (rMin, rMax float64) {
	abs := make([]float64, len(vals))
	for i, v := range vals {
		abs[i] = math.Abs(v)
	}
	sort.Float64s(abs)

	// p-th percentile index (0-indexed, round up).
	p := threshold / 100.0
	idx := max(int(math.Ceil(p*float64(len(abs))))-1, 0)
	idx = min(idx, len(abs)-1)
	clip := abs[idx]
	return -clip, clip
}
