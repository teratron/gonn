package network

import (
	"fmt"

	"github.com/teratron/gonn/pkg/cell"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type Bundle[T utils.Float, S neuron.Nucleus[T]] struct {
	Cells        []S
	_number      uint
	_numberFloat T
}

func NewBundle[T utils.Float, S neuron.Nucleus[T]]() Bundle[T, S] {
	return Bundle[T, S]{
		Cells:        make([]S, 0),
		_number:      0,
		_numberFloat: T(0),
	}
}

func NewBundleWithData[T utils.Float, S neuron.Nucleus[T]](data []T) Bundle[T, S] {
	number := uint(len(data))
	return Bundle[T, S]{
		Cells:        make([]S, 0),
		_number:      number,
		_numberFloat: T(float64(number)),
	}
}

func (b *Bundle[T, S]) Add(cell S) {
	b.Cells = append(b.Cells, cell)
	b._number++
	b._numberFloat = T(float64(b._number))
}

// Bundle for Output or Hidden.
func (b *Bundle[T, S]) GetValues() []*T {
	values := make([]*T, len(b.Cells))
	for i, cell := range b.Cells {
		values[i] = cell.GetValue()
	}
	return values
}

// Bundle for Output or Hidden.
func (b *Bundle[T, S]) GetMisses() []*T {
	misses := make([]*T, len(b.Cells))
	for i, cell := range b.Cells {
		if neuron, ok := any(cell).(neuron.Neuron[T]); ok {
			misses[i] = neuron.GetMiss()
		} else {
			// For non-Neuron types, return nil or zero value
			zero := T(0)
			misses[i] = &zero
		}
	}
	return misses
}

// Input Bundle specific methods
type InputBundle[T utils.Float] struct {
	Bundle[T, *cell.Input[T]]
}

func NewInputBundle[T utils.Float](data []T) InputBundle[T] {
	number := uint(len(data))
	bundle := InputBundle[T]{
		Bundle: Bundle[T, *cell.Input[T]]{
			Cells:        make([]*cell.Input[T], 0),
			_number:      number,
			_numberFloat: T(float64(number)),
		},
	}

	if len(data) > 0 {
		for _, v := range data {
			value := v
			bundle.Cells = append(bundle.Cells, cell.NewInput(&value))
		}
	}

	return bundle
}

// Sets the input data for the network.
func (b *InputBundle[T]) SetInputs(data []T) {
	for i, v := range data {
		if i < len(b.Cells) {
			b.Cells[i].SetValue(&v)
		}
	}
}

// Output Bundle specific methods
type OutputBundle[T utils.Float] struct {
	Bundle[T, *cell.Output[T]]
}

func NewOutputBundle[T utils.Float](data []T) OutputBundle[T] {
	number := uint(len(data))
	bundle := OutputBundle[T]{
		Bundle: Bundle[T, *cell.Output[T]]{
			Cells:        make([]*cell.Output[T], 0),
			_number:      number,
			_numberFloat: T(float64(number)),
		},
	}

	if len(data) > 0 {
		for _, v := range data {
			target := v
			bundle.Cells = append(bundle.Cells, cell.NewOutput(&target))
		}
	}

	return bundle
}

// Sets the target data for the network.
func (b *OutputBundle[T]) SetTargets(data []T) {
	for i, v := range data {
		if i < len(b.Cells) {
			//b.Cells[i].target = &v
			b.Cells[i].SetTarget(&v)
		}
	}
}

// ---------------------------------------------------------
func (b *Bundle[T, S]) SetTargets(data []T) {
	if output, ok := neuron.Nucleus[T](b.Cells[0]).(*cell.Output[T]); ok {
		// Работа с `*cell.Output[T]`
		fmt.Println("Это Output cell", output)
	}
}

// ---------------------------------------------------------
// Hidden Bundle specific methods
type HiddenBundle[T utils.Float] struct {
	Bundle[T, *cell.Hidden[T]]
}

func NewHiddenBundle[T utils.Float](data []T) HiddenBundle[T] {
	number := uint(len(data))
	bundle := HiddenBundle[T]{
		Bundle: Bundle[T, *cell.Hidden[T]]{
			Cells:        make([]*cell.Hidden[T], 0),
			_number:      number,
			_numberFloat: T(float64(number)),
		},
	}

	if len(data) > 0 {
		for range data {
			bundle.Cells = append(bundle.Cells, cell.NewHidden[T]())
		}
	}

	return bundle
}

func (b Bundle[T, S]) Default() Bundle[T, S] {
	return Bundle[T, S]{
		Cells:        make([]S, 0),
		_number:      0,
		_numberFloat: T(0),
	}
}
