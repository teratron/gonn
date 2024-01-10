package rate

import . "github.com/teratron/gonn/pkg"

const DEFAULT = .3

// CheckLearningRate.
func CheckLearningRate[T Floater](rate T) T {
	if rate <= 0 || rate > 1 {
		return T(DEFAULT)
	}

	return rate
}
