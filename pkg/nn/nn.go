//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*nn)(nil)

//
type NeuralNetwork interface {
	//
	Architecture

	//
	Constructor

	//
	GetterSetter
}

//
type nn struct {
	Architecture	`json:"-"`	// Architecture of neural network

	IsInit  bool	// Neural network initializing flag
	IsTrain bool	// Neural network training flag

	/*json	string
	xml		string
	csv		string*/
}