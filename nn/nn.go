//
package nn

type NN interface {
	Initializing()
	Training()
	Querying()
	Verifying()
}

type Specifier interface {
	Perceptron() NeuralNetwork
	FeedForward() NeuralNetwork
	RadialBasis() NeuralNetwork
	Hopfield() NeuralNetwork
	GetterSetter
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

/*type Parameter interface {
	Bias() Bias
}*/

/*

type Neuroner interface {
	Set()
}
type Settings interface {
	Bias() Checker
}

type Processor interface {
}
*/
