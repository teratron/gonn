package network

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
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

func (b *Bundle[T, S]) GetValues() []*T {
	values := make([]*T, len(b.Cells))
	for i, c := range b.Cells {
		values[i] = c.GetValue()
	}
	return values
}

// Bundle for Output or Hidden.

func (b *Bundle[T, S]) GetMisses() []*T {
	misses := make([]*T, len(b.Cells))
	//for i, c := range b.Cells {
	//	if n, ok := any(c).(neuron.Neuron[T]); ok {
	//		misses[i] = n.GetMiss()
	//	} else {
	//		// For non-Neuron types, return nil or zero value
	//		zero := T(0)
	//		misses[i] = &zero
	//	}
	//}
	if c, ok := any(b.Cells).([]neuron.Neuron[T]); ok {
		for i, n := range c {
			misses[i] = n.GetMiss()
		}

	}
	return misses
}

// Input Bundle specific methods

//func NewInputBundle[T utils.Float](data []T) InputBundle[T] {
//	number := uint(len(data))
//	bundle := InputBundle[T]{
//		Bundle: Bundle[T, *cell.Input[T]]{
//			Cells:        make([]*cell.Input[T], 0),
//			_number:      number,
//			_numberFloat: T(float64(number)),
//		},
//	}
//
//	if len(data) > 0 {
//		for _, v := range data {
//			value := v
//			bundle.Cells = append(bundle.Cells, cell.NewInput(&value))
//		}
//	}
//
//	return bundle
//}
//
//func (b *InputBundle[T]) SetInputs(data []T) {
//	for i, v := range data {
//		if i < len(b.Cells) {
//			b.Cells[i].SetValue(&v)
//		}
//	}
//}

// Output Bundle specific methods

//func NewOutputBundle[T utils.Float](data []T) *cell.Output[T] {
//	number := uint(len(data))
//	bundle := OutputBundle[T]{
//		Bundle: Bundle[T, *cell.Output[T]]{
//			Cells:        make([]*cell.Output[T], 0),
//			_number:      number,
//			_numberFloat: T(float64(number)),
//		},
//	}
//
//	if len(data) > 0 {
//		for _, v := range data {
//			target := v
//			bundle.Cells = append(bundle.Cells, cell.NewOutput(&target))
//		}
//	}
//
//	return bundle
//}

func (b *Bundle[T, S]) SetTargets(data *[]T) {
	if o, ok := any(b).(*cell.Output[T]); ok {
		for i, v := range *data {
			if i < len(b.Cells) {
				o.SetTarget(&v)
			}
		}
	}
}

// Hidden Bundle specific methods

//func NewHiddenBundle[T utils.Float](data []T) HiddenBundle[T] {
//	number := uint(len(data))
//	bundle := HiddenBundle[T]{
//		Bundle: Bundle[T, *cell.Hidden[T]]{
//			Cells:        make([]*cell.Hidden[T], 0),
//			_number:      number,
//			_numberFloat: T(float64(number)),
//		},
//	}
//
//	if len(data) > 0 {
//		for range data {
//			bundle.Cells = append(bundle.Cells, cell.NewHidden[T]())
//		}
//	}
//
//	return bundle
//}

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

func (b *Bundle[T, _]) calculateValues() {
	if c, ok := any(b.Cells).([]neuron.Neuron[T]); ok {
		for _, n := range c {
			n.CalculateValue()
		}
	}
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

func (b *Bundle[T, _]) calculateWeights(rate *T) {
	if c, ok := any(b.Cells).([]neuron.Neuron[T]); ok {
		for _, n := range c {
			n.CalculateWeight(rate)
		}
	}
}
