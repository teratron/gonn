package loss

// Categorical Hinge Loss: CAT_HINGE = max(0, 1 - predicted[true_class] + predicted[other_class])
// This is a simplified version that assumes binary classification for now
func catHingeLoss[T float32 | float64](predicted, target T) T {
	// For now, using a simplified approach similar to binary hinge loss
	margin := 1 - predicted*target
	if margin > 0 {
		return margin
	}
	return 0
}
