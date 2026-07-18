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

	// Conv/recurrent/embedding prefixes cache per-call state in their layer
	// structs (lastInput, argmax, …), so a prefixed forward is not read-safe.
	// Serialise those Query calls with the exclusive lock and use the mutating
	// dense path. Regularizer inference masks are no-ops (L1/L2 unchanged,
	// Dropout passes through), so they are skipped.
	if len(n.convPrefix) > 0 {
		n.mu.Lock()
		defer n.mu.Unlock()
		netInput, err := n.runConvForward(input)
		if err != nil {
			return nil, err
		}
		if err := n.SetInputs(netInput); err != nil {
			return nil, err
		}
		n.CalculateValues()
		out := make([]T, n.Network.Output.Len())
		for i, c := range n.Network.Output.Cells() {
			out[i] = *c.GetValue()
		}
		return out, nil
	}

	// Pure dense network: run the stateless forward under a read lock so any
	// number of goroutines may Query in parallel without racing (audit D1).
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.Network.InferDense(input)
}
