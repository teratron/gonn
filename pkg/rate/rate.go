package rate

import (
	"github.com/teratron/gonn/pkg/nn"
)

// CheckLearningRate.
func CheckLearningRate[T nn.Floater](rate T) T {
	if rate <= 0 || rate > 1 {
		return .3
	}
	return rate
}
