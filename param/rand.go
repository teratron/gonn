package param

import (
	"math/rand"
	"time"
)

var GetRandFloat = getRandFloat

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// getRand return random number from -0.5 to 0.5.
func getRandFloat() (r float64) {
	for r == 0 || r > .5 || r < -.5 {
		r = rand.Float64() - .5
	}
	return
}
