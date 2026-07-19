package quantization

import "github.com/teratron/gonn/pkg/utils"

// QuantMode selects the operating mode for a quantization run.
//
// AI-Meta:
//   - Purpose: Enum selecting weight-only vs full-int8 quantization mode.
//   - Implementations: WeightOnly, FullInt8.
type QuantMode int

const (
	// WeightOnly quantizes weights to int8; activations pass through as T.
	// No calibration samples required (QUANT-C1). Phase α.
	WeightOnly QuantMode = iota
	// FullInt8 quantizes both weights and activations using an int32 accumulator GEMM.
	// Requires CalibrationRunner activation statistics. Phase β+.
	FullInt8
)

// QuantizationConfig holds all parameters governing a quantization run.
//
// AI-Meta:
//   - Purpose: Configuration for Quantize[T]; use DefaultQuantizationConfig[T]() for typical weight-only runs.
type QuantizationConfig[T utils.Float] struct {
	Mode                QuantMode
	WeightGranularity   Granularity   // default PerChannel (QUANT-3)
	ActGranularity      Granularity   // PerTensor only in v0.1 (QUANT-C4)
	Strategy            CalibStrategy // default Percentile99p9
	PercentileThreshold float64       // effective when Strategy==Percentile99p9; default 99.9
	MinCalibSamples     int           // hard floor 32; soft floor 100 (QUANT-4)
	Seed                uint64        // RNG seed recorded in CalibrationProvenance
}

// CalibrationProvenance records audit metadata of a calibration run embedded in
// the .qnn.json artifact for traceability (QUANT-6).
//
// AI-Meta:
//   - Purpose: Audit record stored in .qnn.json identifying calibration run parameters.
type CalibrationProvenance struct {
	SampleCount int
	Seed        uint64
	Strategy    CalibStrategy
}

// DefaultQuantizationConfig returns the canonical defaults for a weight-only run:
// Mode=WeightOnly, WeightGranularity=PerChannel, Strategy=Percentile99p9,
// PercentileThreshold=99.9, MinCalibSamples=32.
func DefaultQuantizationConfig[T utils.Float]() QuantizationConfig[T] {
	return QuantizationConfig[T]{
		Mode:                WeightOnly,
		WeightGranularity:   PerChannel,
		ActGranularity:      PerTensor,
		Strategy:            Percentile99p9,
		PercentileThreshold: 99.9,
		MinCalibSamples:     32,
	}
}
