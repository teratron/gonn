package nn

// Default learning rate
const DefaultRate float64 = .3

// checkLearningRate
func checkLearningRate(rate float64) FloatType {
	switch {
	case rate < 0 || rate > 1:
		return FloatType(DefaultRate)
	default:
		return FloatType(rate)
	}
}
