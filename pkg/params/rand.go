package params

import (
	"math/rand"
	"time"

	"github.com/zigenzoog/gonn/pkg"
)

var GetRandFloat = getRandFloat

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// getRandFloat return random number from -0.5 to 0.5.
func getRandFloat() (r pkg.FloatType) {
	for r == 0 {
		r = pkg.FloatType(rand.Float64() - .5)
	}
	return
}
