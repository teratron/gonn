package nn

// Default learning rate
const DefaultRate float32 = .3

// checkLearningRate
func checkLearningRate(rate float32) FloatType {
	switch {
	case rate < 0 || rate > 1:
		return FloatType(DefaultRate)
	default:
		return FloatType(rate)
	}
}
