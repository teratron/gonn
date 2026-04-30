package nn

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Verify runs forward propagation and returns the configured-loss value
// without performing the backward pass or updating weights. Equivalent
// to "what would Train compute if it stopped after the forward step".
//
// Mutates the cell values (forward writes through them) but leaves
// weights and biases untouched — safe to invoke between training
// epochs to monitor validation-set loss without contaminating learning.
//
// Returns ErrInputData on shape mismatch; ErrUserConfig when called on
// a non-Operational network.
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
