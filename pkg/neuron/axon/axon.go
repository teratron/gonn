package axon

import (
	"math/rand"
	"sync"
	"time"

	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type Bundle[T utils.Float] []*Axon[T]

type Axon[T utils.Float] struct {
	Weight T `json:"weight" xml:"weight"`
	//OutgoingCell neuron.Nucleus[T] `json:"-" xml:"-"`
	//CellId [2]uint           `json:"cellId" xml:"cellId"`
	Cell neuron.Nucleus[T] `json:"-" xml:"-"` // Incoming cell: Hidden, Input, Bias
}

var (
	rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	mu  sync.Mutex
)

// New creates a new axon with random weight initialization in range [-0.5, 0.5]
func New[T utils.Float](cell, outgoingCell neuron.Nucleus[T]) *Axon[T] {
	// Create a local random generator with current time as seed
	//rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	mu.Lock()
	//weight := T(rng.Float64()*1.0 - 0.5)
	defer mu.Unlock()
	return &Axon[T]{
		Weight: T(rng.Float64()*1.0 - 0.5), // weight, // Случайное значение в диапазоне [-0.5, 0.5]
		//OutgoingCell: outgoingCell,
		//CellId: cell.GetId(),
		Cell: cell,
	}
}

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION
// ----------------------------------------------------------------------------

func (a *Axon[T]) CalculateValue() T {
	return *a.Cell.GetValue() * a.Weight
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION
// ----------------------------------------------------------------------------

func (a *Axon[T]) CalculateMiss() T {
	return *a.OutgoingCell.GetMiss() * a.Weight
}

func (a *Axon[T]) CalculateWeight(gradient *T) {
	a.Weight += *gradient * *a.Cell.GetValue()
}
