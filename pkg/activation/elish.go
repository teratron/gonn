package activation

import "math"

// ELISH activation function: f(x) = x >= 0 ? x * sigmoid(x) : (exp(x) - 1) * sigmoid(x)
func elishActivation[T float32 | float64](value T) T {
	sigmoidVal := T(1.0) / (T(1.0) + T(math.Exp(float64(-value))))
	if value >= T(0) {
		return value * sigmoidVal
	} else {
		return (T(math.Exp(float64(value))) - T(1.0)) * sigmoidVal
	}
}

// ELISH derivative function
func elishDerivative[T float32 | float64](value T) T {
	sigmoidVal := T(1.0) / (T(1.0) + T(math.Exp(float64(-value))))
	sigmoidDeriv := sigmoidVal * (T(1.0) - sigmoidVal)
	if value >= T(0) {
		// d/dx x * sigmoid(x) = sigmoid(x) + x * sigmoid'(x)
		return sigmoidVal + value*sigmoidDeriv
	} else {
		expVal := T(math.Exp(float64(value)))
		// d/dx (e^x - 1) * sigmoid(x) = e^x * sigmoid(x) + (e^x - 1) * sigmoid'(x)
		// Let's break this down step by step
		term1 := expVal * sigmoidVal
		term2 := (expVal - T(1.0)) * sigmoidDeriv
		return term1 + term2
	}
}
