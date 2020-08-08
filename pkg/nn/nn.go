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

type Tester interface {
	com()
}

type test0 struct {
	Type	Tester
	Map		map[string]Tester
}

type test1 struct {
	Name	string
}

type test2 struct {
	Name2	string
}

func (t test1) com() {}
func (t test2) com() {}