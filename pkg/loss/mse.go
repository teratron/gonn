package loss

// Mean Squared Error loss function: MSE = (predicted - target)²
func mseLoss[T float32 | float64](predicted, target T) T {
	diff := predicted - target
	return diff * diff
}
