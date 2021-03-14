package parameter

import (
	"math/rand"
	"time"
)

// MaxIteration the maximum number of iterations after which training is forcibly terminated.
const MaxIteration int = 10e+05

var (
	GetMaxIteration = getMaxIteration
	GetRandFloat    = getRandFloat
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func getMaxIteration() int {
	return MaxIteration
}

// getRand return random number from -0.5 to 0.5.
func getRandFloat() (r float64) {
	for r == 0 || r > .5 || r < -.5 {
		r = rand.Float64() - .5
	}
	return
}
