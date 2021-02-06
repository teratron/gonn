package nn

// DefaultRate default learning rate
const DefaultRate float64 = .3

// checkLearningRate
func checkLearningRate(rate float64) float64 {
	if rate < 0 || rate > 1 {
		return DefaultRate
	}
	return rate
}
