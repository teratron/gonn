package math

import "math"

// Rounding mode.
const (
	ROUND uint8 = iota // ROUND - round to nearest.
	FLOOR              // FLOOR - round down.
	CEIL               // CEIL - round up.
)

// Round rounding to a floating-point value.
func Round(f float64, mode uint8, prec uint) float64 {
	d := math.Pow(10, float64(prec))
	f *= d
	switch mode {
	case ROUND:
		return math.Round(f) / d
	case FLOOR:
		return math.Floor(f) / d
	case CEIL:
		return math.Ceil(f) / d
	}
	return math.NaN()
}
