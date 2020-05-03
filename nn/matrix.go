package nn

type NN interface {
	Set(Setter)
	Get() Getter
	Check()
}

type Setter interface {
}

type Getter interface {
}

type Checker interface {
}

type Processor interface {
	Init()
	Train()
	Query()
	Test()
}

type ()
