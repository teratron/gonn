package params

import (
	"github.com/teratron/gonn/pkg/nn"
	"math/rand"
)

var GetRandFloat = getRandFloat

// getRandFloat return random number from -0.5 to 0.5.
func getRandFloat() (r nn.FloatType) {
	for r == 0 {
		r = nn.FloatType(rand.Float64() - .5)
	}

	return
}
