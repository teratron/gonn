package nn

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Query runs forward propagation only and returns a freshly allocated
// slice of output values. Read-only with respect to weights — safe to
// invoke concurrently from multiple goroutines once the network is
// Operational.
//
// Returns ErrInputData wrapped with the size mismatch when the supplied
// input length does not match the configured input layer; ErrUserConfig
// wrapped when the network has not been compiled yet.
func (n *NN[T]) Query(input []T) ([]T, error) {
	if n.stateField != stateOperational {
		return nil, utils.Newf(utils.ErrUserConfig,
			"Query: network is %s, must be Operational (call Compile or use New)", n.stateField.String())
	}
	if err := n.SetInputs(input); err != nil {
		return nil, err
	}
	n.CalculateValues()

	out := make([]T, n.Network.Output.Len())
	for i, c := range n.Network.Output.Cells() {
		out[i] = *c.GetValue()
	}
	return out, nil
}
