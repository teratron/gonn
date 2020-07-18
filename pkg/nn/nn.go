//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*nn)(nil)

//
type NeuralNetwork interface {
	//
	Architecture

	//
	Processor

	//
	GetterSetter
}

//
type nn struct {
	Architecture			// Architecture of neural network

	isInit		bool		// Neural network initializing flag
	isQuery		bool		// Neural network querying flag
	isTrain		bool		// Neural network training flag

	language	langType
	logging		modeLogType
}