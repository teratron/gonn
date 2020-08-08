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


}

//
type nn struct {
	Architecture		`json:"architecture"`	// Architecture of neural network

	isInit  bool								// Neural network initializing flag
	IsTrain bool		`json:"isTrain"`		// Neural network training flag

	json	jsonType
	xml		xmlType
	csv		csvType
	db		dbType
}

type NN struct {
	*nn
}