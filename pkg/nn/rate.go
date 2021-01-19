package nn

// DefaultRate default learning rate
const DefaultRate float64 = .3

// checkLearningRate
func checkLearningRate(rate float64) float64 {
	switch {
	case rate < 0 || rate > 1:
		return DefaultRate
	default:
		return rate
	}
}
