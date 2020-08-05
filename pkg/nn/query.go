//
package nn

import "github.com/zigenzoog/gonn/pkg"

func (n *NN) Query(input []float64) (output []float64) {
	if !n.IsTrain {
		pkg.Log("Neural network is not trained", true) // !!!
		if !n.IsInit {
			pkg.Log("Error initialization", true) // !!!
			return nil
		}
	}
	if a, ok := n.Get().(NeuralNetwork); ok {
		output = a.Query(input)
	}
	return
}