package nn

// DefaultRate default learning rate
const DefaultRate float32 = .3

// checkLearningRate
func checkLearningRate(rate float32) float32 {
	switch {
	case rate < 0 || rate > 1:
		return DefaultRate
	default:
		return rate
	}
}
