package math

import "math"

// Rounded mode.
const (
	ModeRound uint8 = iota // Round (round to nearest).
	ModeFloor              // Floor (round down).
	ModeCeil               // Ceil (round up).
)

// Round rounding to a floating-point value.
func Round(f float64, mode uint8, prec uint) float64 {
	d := math.Pow(10, float64(prec))
	f *= d
	switch mode {
	case ModeRound:
		return math.Round(f) / d
	case ModeFloor:
		return math.Floor(f) / d
	case ModeCeil:
		return math.Ceil(f) / d
	}
	return math.NaN()
}
