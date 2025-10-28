package rate

import . "github.com/teratron/gonn/pkg"

const DEFAULT = .3

// CheckLearningRate.
func CheckLearningRate[T Floater](value T) T {
	if value <= 0 || value > 1 {
		return T(DEFAULT)
	}

	return value
}
