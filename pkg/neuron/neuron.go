package neuron

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Cell type tags identify the role of a neural cell within the network graph.
// The tag occupies the first element of every concrete cell's 2-tuple identity.
//
// AI-Meta:
//   - Purpose: Cell role constants for identity tuples and network introspection.
const (
	UNKNOWN uint8 = iota // UNKNOWN — zero value; indicates an uninitialized cell type.
	INPUT                // INPUT — cell belongs to the input layer; carries an externally supplied feature.
	OUTPUT               // OUTPUT — cell belongs to the output layer; participates in loss computation.
	DENSE                // DENSE — cell belongs to a hidden (dense) layer; owns incoming axons.
	BIAS                 // BIAS — bias unit injected by non-input layers when bias is enabled.
)

// Nucleus is the base contract shared by every neural cell type.
// It provides read access to a cell's scalar value via GetValue.
//
// AI-Meta:
//   - Purpose: Minimal read-only interface shared by every cell role (Input, Bias, Dense, Output).
//   - Implementations: [cell.Input], [cell.Bias], [cell.Dense], [cell.Output].
//   - Related: [Neuron].
type Nucleus[T utils.Float] interface {
	// GetValue returns the current value of the cell
	GetValue() *T
	//SetValue(T)
}

// Neuron extends Nucleus with backward-pass operations: error access and weight update.
// Implemented by cells that participate in gradient descent (dense and output cells).
//
// AI-Meta:
//   - Purpose: Full training interface for hidden and output cells that perform backprop.
//   - Implementations: [cell.Dense], [cell.Output].
//   - Related: [Nucleus].
type Neuron[T utils.Float] interface {
	Nucleus[T]

	// GetMiss returns the error (difference between target and obtained value)
	GetMiss() *T

	SetMiss(T)

	// CalculateValue calculates the neuron value based on input signals
	CalculateValue()

	// CalculateWeight calculates the neuron weight based on error
	CalculateWeight(*T)
}
