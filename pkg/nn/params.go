package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
)

// Activation function mode.
const (
	LINEAR    = activation.LINEAR    // LINEAR - Linear/identity.
	RELU      = activation.RELU      // RELU - ReLu (rectified linear unit).
	LEAKYRELU = activation.LEAKYRELU // LEAKYRELU - Leaky ReLu (leaky rectified linear unit).
	SIGMOID   = activation.SIGMOID   // SIGMOID - Logistic, a.k.a. sigmoid or soft step.
	TANH      = activation.TANH      // TANH - TanH (hyperbolic tangent).
)

// The mode of calculation of the total error.
const (
	MSE    = loss.MSE    // MSE - Mean Squared Error.
	RMSE   = loss.RMSE   // RMSE - Root Mean Squared Error.
	ARCTAN = loss.ARCTAN // ARCTAN - Arctan Error.
	AVG    = loss.AVG    // AVG - Average Error.
)
