package nn

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Query runs forward propagation only and returns a freshly allocated slice
// of output values. Weights are never modified — safe to call concurrently
// from multiple goroutines once the network is Operational.
//
// AI-Meta:
//   - Purpose: Run inference on one input vector; does not update weights.
//   - Usage: out, err := n.Query([]float32{0.1, 0.2, 0.3}).
//   - Concurrency: ReadSafe; multiple goroutines may call Query simultaneously.
//   - Errors: ErrUserConfig (not Operational), ErrInputData (length mismatch).
//   - Related: [Train], [Verify], [NN].
//   - Stability: Stable.
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
