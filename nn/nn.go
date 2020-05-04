package nn

type NN interface {
	Init()
	Train()
	Query()
	Test()
	Getter
	Setter
}

type Getter interface {
	Get() float32
}

type Setter interface {
	Set()
}

type Checker interface {
	Check()
}

type Processor interface{}
type Settings interface{}
