package loss

// Mean Absolute Error loss function: MAE = |predicted - target|
func maeLoss[T float32 | float64](predicted, target T) T {
	diff := predicted - target
	if diff < 0 {
		return -diff
	}
	return diff
}
