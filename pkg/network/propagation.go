package network

import (
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// ----------------------------------------------------------------------------
// Forward propagation methods
// ----------------------------------------------------------------------------

// CalculateValues calculates the value of all neurons in the network
func (n *Network[T]) calculateValues() {
	for _, neuron := range n.Cells {
		neuron.CalculateValue()
	}
}

// CalculateLoss calculates and returns the total error of the output neurons
func (n *Network[T]) calculateLoss() T {
	return loss.CalculateTotalLoss(n.Output.GetMisses(), n.Loss)
}

// ----------------------------------------------------------------------------
// Backward propagation methods
// ----------------------------------------------------------------------------

// CalculateMisses calculates the error of hidden neurons
// Implements backward propagation by processing neurons in reverse order
func (n *Network[T]) calculateMisses() *Network[T] {
	// Process hidden neurons in reverse order for backpropagation
	cells := n.Hidden.Cells
	for i := len(cells) - 1; i >= 0; i-- {
		calculateMissForHidden(cells[i])
	}
	return n
}

// Helper function to calculate miss for a hidden neuron
// This simulates the calculate_miss() method from the Rust implementation
func calculateMissForHidden[T utils.Float](neuron *cell.Hidden[T]) {
	// Accumulate error from each outgoing connection using the axon's CalculateMiss method logic
	// In backpropagation, the error contribution from each outgoing axon is:
	// error_from_next_layer * weight_of_connection
	var cum T = 0.0
	for _, axon := range neuron.OutgoingAxons {
		// Calculate the error contribution from this axon to the current neuron
		cum += axon.CalculateMiss()

	}
	neuron.SetMiss(cum)
}

// CalculateWeights updates weights of all neurons in the network
func (n *Network[T]) calculateWeights() {
	// Update weights for all neurons in the network
	for _, neuron := range n.Cells {
		neuron.CalculateWeight(&n.Rate)
	}
}
