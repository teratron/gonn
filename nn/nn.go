//
package nn

type NN interface {
	Init()
	Train()
	Query()
	Test()
}

type Architecture interface {
	Perceptron() NeuralNetwork
	FeedForward() NeuralNetwork
	RadialBasis() NeuralNetwork
	Hopfield() NeuralNetwork
}

/*type Neuroner interface {
	Set()
}
*/
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

/*
type Checker interface {
	Check()
}

type Settings interface {
	Bias() Checker
}

type Processor interface {
}
*/
