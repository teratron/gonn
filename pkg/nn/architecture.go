//
package nn

//import "github.com/zigenzoog/gonn/pkg/nn/architecture/hopfield"

type Architecture interface {
	//
	//perceptron() NeuralNetwork


	//
	//RadialBasis() NeuralNetwork

	//
	//Hopfield() NeuralNetwork
	//Hopfield

	//
	Parameter
}

/*type Hopfield interface {
	Hopfield() NeuralNetwork
}*/

/*type ReadWriter interface {
	io.ReadWriter
	Architecture
}*/


/*func (n *NN) Hopfield() NeuralNetwork {
	n.Architecture = &hopfield.Hopfield{
		Architecture: n,
	}
	return n
}*/
