package loss

// Hinge Loss: HINGE = max(0, 1 - predicted * target)
// This assumes target is -1 or 1, and predicted is the raw output from the model
func hingeLoss[T float32 | float64](predicted, target T) T {
	margin := 1 - predicted*target
	if margin > 0 {
		return margin
	}
	return 0
}
