package nn

import "github.com/zigenzoog/gonn/pkg"

// Architecture
type Architecture interface {
	architecture() Architecture
	setArchitecture(Architecture)
	pkg.GetSetter
}

/*type architecture struct {
	Architecture
}*/

func (n *nn) architecture() Architecture {
	return n.Architecture
}

func (n *nn) setArchitecture(network Architecture) {
	n.Architecture = network
	n.Architecture.setArchitecture(n)
}
