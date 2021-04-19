package nn

import "github.com/teratron/gonn/pkg/params"

const (
	// Activation function mode.
	ModeLINEAR    = params.ModeLINEAR    // Linear/identity.
	ModeRELU      = params.ModeRELU      // ReLu (rectified linear unit).
	ModeLEAKYRELU = params.ModeLEAKYRELU // Leaky ReLu (leaky rectified linear unit).
	ModeSIGMOID   = params.ModeSIGMOID   // Logistic, a.k.a. sigmoid or soft step.
	ModeTANH      = params.ModeTANH      // TanH (hyperbolic tangent).

	// The mode of calculation of the total error.
	ModeMSE    = params.ModeMSE    // Mean Squared Error.
	ModeRMSE   = params.ModeRMSE   // Root Mean Squared Error.
	ModeARCTAN = params.ModeARCTAN // Arctan.
	ModeAVG    = params.ModeAVG

	// DefaultRate default learning rate.
	DefaultRate = params.DefaultRate
)
