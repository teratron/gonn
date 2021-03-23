package param

// DefaultRate default learning rate
const DefaultRate float64 = .3

// CheckLearningRate
func CheckLearningRate(rate float64) float64 {
	if rate < 0 || rate > 1 {
		return DefaultRate
	}
	return rate
}
