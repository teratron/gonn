package nn

// Default learning rate
const DefaultRate float64 = .3

// checkLearningRate
func checkLearningRate(rate float64) floatType {
	switch {
	case rate < 0 || rate > 1:
		return floatType(DefaultRate)
	default:
		return floatType(rate)
	}
}
