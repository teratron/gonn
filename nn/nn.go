package nn

type NN interface {
	Init()
	Train()
	Query()
	Test()
}

type Typer interface {
	Perceptron() NeuralNetwork
	FeedForward() NeuralNetwork
	RadialBasis() NeuralNetwork
	Hopfield() NeuralNetwork
}

type Neuroner interface {
	Set()
}

/*
type GetterSetter interface {
	Get() float64
	Set()
}

type Getter interface {
	Get() float64
}

type Setter interface {
	Set()
}

type Checker interface {
	Check()
}

type Settings interface{
	Bias() Checker
}

type Processor interface{
}
*/
