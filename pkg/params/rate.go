package params

import "github.com/teratron/gonn/pkg"

// CheckLearningRate.
func CheckLearningRate(rate pkg.FloatType) pkg.FloatType {
	if rate <= 0 || rate > 1 {
		return .3
	}
	return rate
}
