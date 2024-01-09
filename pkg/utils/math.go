package utils

import (
	"math"

	"github.com/teratron/gonn/pkg"
)

// Pow returns x**y, the base-x exponential of y.
func Pow[T pkg.Floater](x T, y float64) T {
	switch v := any(x).(type) {
	case float32:
		return T(math.Pow(float64(v), y))
	case float64:
		return T(math.Pow(v, y))
	default:
		panic(v) // TODO:
	}
}

func Pow2[T pkg.Floater](x T, y float64) T {
	return T(math.Pow(float64(x), y))
}

func Exp[T pkg.Floater](x T) T {
	switch v := any(x).(type) {
	case float32:
		return T(math.Exp(float64(v)))
	case float64:
		return T(math.Exp(v))
	default:
		panic(v) // TODO:
	}
}

func IsNaN[T pkg.Floater](f T) bool {
	switch v := any(f).(type) {
	case float32:
		return math.IsNaN(float64(v))
	case float64:
		return math.IsNaN(v)
	default:
		return false
	}
}

func IsInf[T pkg.Floater](f T, sign int) bool {
	switch v := any(f).(type) {
	case float32:
		return math.IsInf(float64(v), sign)
	case float64:
		return math.IsInf(v, sign)
	default:
		return false
	}
}

func Round[T pkg.Floater](x T, precision uint) T {
	d := Pow[T](10, float64(precision))
	x *= d
	switch v := any(x).(type) {
	case float32:
		return T(math.Round(float64(v))) / d
	case float64:
		return T(math.Round(v)) / d
	default:
		panic(v) // TODO:
	}
}

func Floor[T pkg.Floater](x T, precision uint) T {
	d := Pow[T](10, float64(precision))
	x *= d
	switch v := any(x).(type) {
	case float32:
		return T(math.Floor(float64(v))) / d
	case float64:
		return T(math.Floor(v)) / d
	default:
		panic(v) // TODO:
	}
}

func Ceil[T pkg.Floater](x T, precision uint) T {
	d := Pow[T](10, float64(precision))
	x *= d
	switch v := any(x).(type) {
	case float32:
		return T(math.Ceil(float64(v))) / d
	case float64:
		return T(math.Ceil(v)) / d
	default:
		panic(v) // TODO:
	}
}

/*type Caller interface {
	func(float64) float64 | func(float64, float64) float64
	// | func(float64) bool | func(float64, int) bool
}

func to[T pkg.Floater, U Caller](value T, call U) T {
	switch v := any(value).(type) {
	case float32:

		return T(call(float64(v)))
	case float64:
		return T(call(v))
	default:
		panic(v) // TODO:
	}
}*/
