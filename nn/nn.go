package nn

type NN interface {
	Init()
	Train()
	Query()
	Test()
	//Getter
	//Setter
}

type Neuroner interface {
	Set()
}

/*type Getter interface {
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

type Processor interface{}*/