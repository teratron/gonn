// Radial Basis Neural Network - under construction
package nn

import "fmt"

/*type RadialBasis interface {
	RadialBasis() NeuralNetwork
}*/

type radialBasis struct {
	Architecture
	Processor

	cup int

	neuron struct {
		error		floatType
	}
}

// Initializing Radial Basis Neural Network
func (n *nn) RadialBasis() NeuralNetwork {
	n.architecture = &radialBasis{
		Architecture:	n,
	}
	return n
}

// Setter
func (r *radialBasis) Set(set ...Setter) {
	switch v := set[0].(type) {
	/*case BiasType:
		r.bias = v
	case RateType:
		r.rate = v
	case HiddenType:
		r.hiddenLayer = v*/
	default:
		fmt.Printf("%T %v\n", v, v)
		Log("This type of variable is missing for Radial Basis Neural Network", false)
	}
}

// Getter
func (r *radialBasis) Get(set ...Setter) Getter {
	switch set[0].(type) {
	/*case Bias:
		return r.bias
	case Rate:
		return r.rate
	case Hidden:
		return r.hiddenLayer*/
	default:
		Log("This type of variable is missing for Perceptron Neural Network", false)
		return nil
	}
}

// Train
/*func (r *radialBasis) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (r *radialBasis) Query(input []float64) []float64 {
	panic("implement me")
}*/