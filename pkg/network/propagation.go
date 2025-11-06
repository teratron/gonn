package network

import (
	"github.com/teratron/gonn/pkg/loss"
)

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

func (n *Network[T]) calculateValues() {
	n.Hidden.calculateValues()
	n.Output.calculateValues()
}

func (n *Network[T]) calculateLoss(mode loss.Type) T {
	return loss.CalculateTotalLoss(n.Output.GetMisses(), mode)
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

func (n *Network[T]) calculateMisses() *Network[T] {
	for i := n.Hidden.number - 1; i >= 0; i-- {
		var cum T = 0.0
		for _, a := range n.Hidden.Cells[i].OutgoingAxons {
			cum += a.CalculateMiss()
		}
	}
	return n
}

func (n *Network[T]) calculateWeights(rate *T) {
	n.Hidden.calculateWeights(rate)
	n.Output.calculateWeights(rate)
}
