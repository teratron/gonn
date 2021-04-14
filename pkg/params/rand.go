package params

import (
	"math/rand"
	"time"

	"github.com/teratron/gonn/pkg"
)

var GetRandFloat = getRandFloat

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// getRand return random number from -0.5 to 0.5
func getRandFloat() (r pkg.FloatType) {
	for r == 0 || r > .5 || r < -.5 {
		r = pkg.FloatType(rand.Float64() - .5)
	}
	return
}
