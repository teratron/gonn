package network

import (
	"github.com/teratron/gonn/pkg/loss"
)

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

func (n *Network[T]) CalculateValues() {
	n.Hidden.calculateValues()
	n.Output.calculateValues()
}

func (n *Network[T]) CalculateLoss(mode loss.Type) T {
	return loss.CalculateTotalLoss(n.Output.GetMisses(), mode)
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

func (n *Network[T]) CalculateMisses() {
	for i := n.Hidden.size - 1; i >= 0; i-- {
		var cum T = 0.0
		for _, a := range n.Hidden.cells[i].OutgoingAxons {
			cum += a.CalculateMiss()
		}
		n.Hidden.cells[i].SetMiss(cum)
	}
}

func (n *Network[T]) CalculateWeights(rate *T) {
	n.Hidden.calculateWeights(rate)
	n.Output.calculateWeights(rate)
}
