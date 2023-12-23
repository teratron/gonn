package rate

import (
	"github.com/teratron/gonn/pkg"
)

// CheckLearningRate.
func CheckLearningRate[T pkg.Floater](rate T) T {
	if rate <= 0 || rate > 1 {
		return .3
	}

	return rate
}
