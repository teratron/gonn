package params

import (
	"github.com/teratron/gonn/pkg"
)

// DefaultRate default learning rate
const DefaultRate float64 = .3

// CheckLearningRate
func CheckLearningRate(rate pkg.FloatType) pkg.FloatType {
	if rate < 0 || rate > 1 {
		return pkg.FloatType(DefaultRate)
	}
	return rate
}
