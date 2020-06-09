//
package nn

//
type NeuralNetwork interface {
	//
	Perceptron() NeuralNetwork

	//
	FeedForward() NeuralNetwork

	//
	RadialBasis() NeuralNetwork

	//
	Hopfield() NeuralNetwork

	GetterSetter
}

//
type NN interface {
	// Initializing
	Init()

	// Training
	Train()

	// Querying
	Query()

	// Verifying
	Verify()
}

type GetterSetter interface {
	Getter
	Setter
}

type Getter interface {
	Get() Getter
}

type Setter interface {
	Set(Setter)
}

type Checker interface {
	Check() Checker
}

/*
type Parameter interface {
	Bias() bias
}
*/
/*
type Neuroner interface {
	Set()
}
type Settings interface {
	Bias() Checker
}

type Processor interface { //manipulator
}
*/
