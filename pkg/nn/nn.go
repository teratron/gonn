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
	Architecture	`json:"architecture"`	// Architecture of neural network

	isInit  bool	// Neural network initializing flag
	isTrain bool	// Neural network training flag

	/*json	string
	xml		string
	csv		string*/
}