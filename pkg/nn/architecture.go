package nn

import "github.com/zigenzoog/gonn/pkg"

// Architecture
type Architecture interface {
	architecture() Architecture
	setArchitecture(Architecture)
	pkg.GetSetter
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
