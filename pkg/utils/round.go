package utils

import (
	"math"

	"github.com/teratron/gonn/pkg"
)

// Rounding mode.
const (
	ROUND uint8 = iota // ROUND - round to nearest.
	FLOOR              // FLOOR - round down.
	CEIL               // CEIL - round up.
)

// Round rounding to a floating-point value.
func Round2[T pkg.Floater](f T, mode uint8, precision uint) T {
	d := Pow[T](10, float64(precision))
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
