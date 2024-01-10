package utils

import (
	"math/rand"

	. "github.com/teratron/gonn/pkg"
)

var getRandFloat = GetRandFloat[float32]

// GetRandFloat return random number from -0.5 to 0.5.
func GetRandFloat[T Floater]() (r T) {
	for r == 0 {
		r = T(rand.Float64() - .5)
	}

	return
}
