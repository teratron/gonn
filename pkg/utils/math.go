package utils

import (
	"math"

	"github.com/teratron/gonn/pkg"
)

func Pow[T pkg.Floater](x T, y float64) T {
	switch v := any(x).(type) {
	case float32:
		return T(math.Pow(float64(v), y))
	case float64:
		return T(math.Pow(v, y))
	default:
		panic(x) // TODO:
	}
}

func Exp[T pkg.Floater](x T) T {
	switch v := any(x).(type) {
	case float32:
		return T(math.Exp(float64(v)))
	case float64:
		return T(math.Exp(v))
	default:
		panic(x) // TODO:
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
