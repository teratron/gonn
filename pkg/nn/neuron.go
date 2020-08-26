package nn

import "github.com/zigenzoog/gonn/pkg"

// Specifier
type Specifier interface {
	pkg.GetSetter
}

// neuron
type neuron struct {
	Synapser

	value    floatType	// Neuron value
	axon     []*axon	// All incoming axons
	specific Specifier
}