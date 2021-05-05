package nn

import "github.com/teratron/gonn/pkg/params"

const (
	// Activation function mode.
	LINEAR    = params.LINEAR    // Linear/identity.
	RELU      = params.RELU      // ReLu (rectified linear unit).
	LEAKYRELU = params.LEAKYRELU // Leaky ReLu (leaky rectified linear unit).
	SIGMOID   = params.SIGMOID   // Logistic, a.k.a. sigmoid or soft step.
	TANH      = params.TANH      // TanH (hyperbolic tangent).

	// The mode of calculation of the total error.
	MSE    = params.MSE    // Mean Squared Error.
	RMSE   = params.RMSE   // Root Mean Squared Error.
	ARCTAN = params.ARCTAN // Arctan.
	AVG    = params.AVG    // Average.
)
