package loss

// Mean Absolute Percentage Error loss function: MAPE = |(target - predicted) / target| * 100
// To avoid division by zero, we add a small epsilon to the denominator
func mapeLoss[T float32 | float64](predicted, target T) T {
	epsilon := T(1e-7)
	denominator := target
	if target < 0 {
		denominator = -target
	}
	if denominator < epsilon {
		denominator = epsilon
	}

	percentageError := (target - predicted) / denominator
	if percentageError < 0 {
		percentageError = -percentageError
	}
	return percentageError * 100
}
