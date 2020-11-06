package nn

import "github.com/zigenzoog/gonn/pkg"

// Neuroner
type Neuroner interface {
	pkg.GetSetter
}

// neuron
type neuron struct {
	value    floatType // Neuron value
	axon     []*axon   // All incoming axons
	specific Neuroner  // Specific option of neuron: miss (error) or other

	Synapser
}
