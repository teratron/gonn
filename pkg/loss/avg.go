package loss

// Average Error (Mean Absolute Error) loss function: AVG = |predicted - target|
func avgLoss[T float32 | float64](predicted, target T) T {
	return maeLoss(predicted, target)
}
