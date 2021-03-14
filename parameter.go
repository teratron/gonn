package gonn

// Parameter
type Parameter interface {
	NameNN() string
	SetNameNN(string)

	InitNN() bool
	SetInitNN(bool)

	NameJSON() string
	SetNameJSON(string)

	NameYAML() string
	SetNameYAML(string)

	Weight() Floater
	SetWeight(Floater)
}
