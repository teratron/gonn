package params

import (
	"math/rand"

	"github.com/teratron/gonn/pkg"
)

var GetRandFloat = getRandFloat

// getRandFloat return random number from -0.5 to 0.5.
func getRandFloat() (r pkg.FloatType) {
	for r == 0 {
		r = pkg.FloatType(rand.Float64() - .5)
	}

	return
}
