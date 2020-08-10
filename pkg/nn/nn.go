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
	Architecture						`json:"architecture,omitempty"`				// Architecture of neural network
	//Parameter map[string]interface{}	`json:"architecture"`
	//Parameters interface{}	`json:"architecture"`
	isInit    bool                    // Neural network initializing flag
	IsTrain   bool                    `json:"isTrain"`		// Neural network training flag
	json		jsonType
	xml			xmlType
	csv			csvType
	db			dbType
}

//type Parameters interface{}

type NN struct {
	*nn
}

/*type Tester interface {
	com()
}

type test0 struct {
	//Type	Tester
	Architecture		map[string]Tester
}

type test1 struct {
	Name	string
}

type test2 struct {
	Name2	string
}

func (t test1) com() {}
func (t test2) com() {}*/