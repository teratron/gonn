package nn

import "fmt"

// Query
func (n *nn) Query(input []float64) (output []float64) {
	if !n.IsTrain {
		errNN(fmt.Errorf("query: %w", ErrNotTrained))
		if !n.IsInit {
			errNN(fmt.Errorf("%w for query", ErrInit))
			return nil
		}
	}
	if a, ok := n.Architecture.(NeuralNetwork); ok {
		output = a.Query(input)
	}
	return
}