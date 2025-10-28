package loss

// Huber Loss: HUBER = { 0.5 * (predicted - target)²                    if |predicted - target| <= delta
//
//	{ delta * |predicted - target| - 0.5 * delta²    otherwise
//
// Using delta = 1.0 as default
func huberLoss[T float32 | float64](predicted, target T) T {
	return huberLossWithDelta(predicted, target, 1.0)
}

// Huber Loss with custom delta parameter
func huberLossWithDelta[T float32 | float64](predicted, target T, delta float64) T {
	diff := predicted - target
	absDiff := diff
	if diff < 0 {
		absDiff = -diff
	}

	d := T(delta)

	if absDiff <= d {
		return 0.5 * diff * diff
	} else {
		return d*absDiff - 0.5*d*d
	}
}
