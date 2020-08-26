//
package nn

type Architecture interface {
	architecture() Architecture
	setArchitecture(Architecture)
}

type architecture struct {
	Architecture
}

func (n *NN) architecture() Architecture {
	return n.Architecture
}

func (n *NN) setArchitecture(network Architecture) {
	n.Architecture = network
}