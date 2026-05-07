package nn

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Verify runs a forward pass and returns the configured loss without
// performing backprop or updating weights. Cell values are mutated but
// weights and biases are untouched — safe to call between training epochs
// to monitor validation-set loss without contaminating learning.
//
// AI-Meta:
//   - Purpose: Compute validation loss for one sample without touching weights.
//   - Usage: loss, err := n.Verify(validInput, validTarget).
//   - Concurrency: NotSafe; mutates cell values (forward pass).
//   - Errors: ErrUserConfig (not Operational), ErrInputData (shape mismatch).
//   - Related: [Query], [Train], [network.Network.CalculateValues].
//   - Stability: Stable.
func (n *NN[T]) Verify(input, target []T) (T, error) {
	if n.stateField != stateOperational {
		return 0, utils.Newf(utils.ErrUserConfig,
			"Verify: network is %s, must be Operational", n.stateField.String())
	}
	if err := n.SetInputs(input); err != nil {
		return 0, err
	}
	if err := n.SetTargets(target); err != nil {
		return 0, err
	}
	n.CalculateValues()
	return n.CalculateLossDefault(), nil
}
