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
func (n *nn) architecture() Architecture {
	return n.Architecture
}

// setArchitecture
func (n *nn) setArchitecture(network Architecture) {
	n.Architecture = network
}