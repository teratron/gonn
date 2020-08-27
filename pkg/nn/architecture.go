package nn

// Architecture
type Architecture interface {
	architecture() Architecture
	setArchitecture(Architecture)
}

// architecture
type architecture struct {
	Architecture
}

// architecture
func (n *NN) architecture() Architecture {
	return n.Architecture
}

// setArchitecture
func (n *NN) setArchitecture(network Architecture) {
	n.Architecture = network
}