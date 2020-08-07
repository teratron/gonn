//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*net)(nil)

//
type NeuralNetwork interface {
	//
	Architecture

	//
	Constructor
}

//
type net struct {
	Architecture		`json:"architecture"`	// Architecture of neural network

	isInit  bool								// Neural network initializing flag
	IsTrain bool		`json:"isTrain"`		// Neural network training flag

	json	jsonType
	xml		xmlType
	csv		csvType
	db		dbType
}

type NN struct {
	net *net
}