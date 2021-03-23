package nn

import "github.com/teratron/gonn/param"

const (
	// Activation function mode.
	ModeLINEAR    = param.ModeLINEAR    // Linear/identity.
	ModeRELU      = param.ModeRELU      // ReLu (rectified linear unit).
	ModeLEAKYRELU = param.ModeLEAKYRELU // Leaky ReLu (leaky rectified linear unit).
	ModeSIGMOID   = param.ModeSIGMOID   // Logistic, a.k.a. sigmoid or soft step.
	ModeTANH      = param.ModeTANH      // TanH (hyperbolic tangent).

	// The mode of calculation of the total error.
	ModeMSE    = param.ModeMSE    // Mean Squared Error.
	ModeRMSE   = param.ModeRMSE   // Root Mean Squared Error.
	ModeARCTAN = param.ModeARCTAN // Arctan.

	// DefaultRate default learning rate.
	DefaultRate = param.DefaultRate
)
