package nn

import "github.com/teratron/gonn/pkg/params"

// Activation function mode.
const (
	LINEAR    = params.LINEAR    // LINEAR - Linear/identity.
	RELU      = params.RELU      // RELU - ReLu (rectified linear unit).
	LEAKYRELU = params.LEAKYRELU // LEAKYRELU - Leaky ReLu (leaky rectified linear unit).
	SIGMOID   = params.SIGMOID   // SIGMOID - Logistic, a.k.a. sigmoid or soft step.
	TANH      = params.TANH      // TANH - TanH (hyperbolic tangent).
)

// The mode of calculation of the total error.
const (
	MSE    = params.MSE    // MSE - Mean Squared Error.
	RMSE   = params.RMSE   // RMSE - Root Mean Squared Error.
	ARCTAN = params.ARCTAN // ARCTAN - Arctan Error.
	AVG    = params.AVG    // AVG - Average Error.
)
