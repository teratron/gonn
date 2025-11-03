package nn

import (
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
	//"github.com/teratron/gonn/pkg/network"
)

func (n *NN[T]) calculateValues() {
    for _, cell := range n.Network.Cells {
        cell.CalculateValue()
    }
}

func (n *NN[T]) calculateLoss() {
    loss.CalculateTotalLoss(n.Network.Output.Cells.GetMisses(), n.Loss)
}

/*
use super::loss::get_total_loss;
use super::{Float, Rustunumic};
use tracing::{trace};

impl<T: Float> Rustunumic<'_, T> {
    //////////////////////////////////////////////////////////////////////////
    // Forward propagation.
    //////////////////////////////////////////////////////////////////////////

    /// Calculating neuron's value.
    pub(super) fn calculate_values(&mut self) {
        self.network
            .cells
            .iter_mut()
            .for_each(|n| n.calculate_value())
    }

    /// Calculating and return the total error of the output neurons.
    pub(super) fn calculate_loss(&self) -> T {
        get_total_loss(self.network.output.get_misses(), &self.loss_mode)
    }

    //////////////////////////////////////////////////////////////////////////
    // Backward propagation.
    //////////////////////////////////////////////////////////////////////////

    /// Calculating the error of neuron.
    pub(super) fn calculate_misses(&mut self) -> &mut Self {
        self.network
            .hidden
            .cells
            .iter_mut()
            .rev()
            .for_each(|n| n.calculate_miss());
        self
    }

    /// Update weights.
    pub(super) fn calculate_weights(&mut self) {
        trace!("Updating weights for {} cells", self.network.cells.len());
        self.network
            .cells
            .iter_mut()
            .for_each(|n| n.calculate_weight(&self.rate));
        trace!("Weight update completed");
    }
}
*/