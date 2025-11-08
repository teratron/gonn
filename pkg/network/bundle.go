package network

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

type bundle[T utils.Float, S neuron.Nucleus[T]] struct {
	cells       []S
	cache       []*T
	number      int
	numberFloat T
}

func newBundle[T utils.Float, S neuron.Nucleus[T]]() bundle[T, S] {
	return bundle[T, S]{
		cells:       make([]S, 0),
		cache:       make([]*T, 0),
		number:      0,
		numberFloat: T(0),
	}
}

//func newBundleWithData[T utils.Float, S neuron.Nucleus[T]](data []T) bundle[T, S] {
//	number := len(data)
//	return bundle[T, S]{
//		cells:       make([]S, 0),
//		number:      number,
//		numberFloat: T(float64(number)),
//	}
//}

func (b *bundle[T, S]) Add(cell S) {
	b.cells = append(b.cells, cell)
	b.number++
	b.numberFloat = T(b.number)
}

func (b *bundle[T, S]) GetValues() *[]*T {
	//values := make([]*T, len(b.cells))
	for i, c := range b.cells {
		//values[i] = c.GetValue()
		b.cache[i] = c.GetValue()
	}
	return &b.cache
}

// Input bundle specific methods

//	func NewInputBundle[T utils.Float](data []T) InputBundle[T] {
//		number := uint(len(data))
//		bundle := InputBundle[T]{
//			bundle: bundle[T, *cell.Input[T]]{
//				cells:        make([]*cell.Input[T], 0),
//				number:      number,
//				numberFloat: T(float64(number)),
//			},
//		}
//
//		if len(data) > 0 {
//			for _, v := range data {
//				value := v
//				bundle.cells = append(bundle.cells, cell.NewInput(&value))
//			}
//		}
//
//		return bundle
//	}

func (b *bundle[T, _]) SetInputs(data *[]T) {
	if len(*data) > len(b.cells) {
		utils.Logger.Error("data length is greater than bundle length")
	}
	if i, ok := any(b).(*cell.Input[T]); ok {
		for _, v := range *data {
			i.SetValue(&v)
		}
	}
}

// Output bundle specific methods

//func NewOutputBundle[T utils.Float](data []T) *cell.Output[T] {
//	number := uint(len(data))
//	bundle := OutputBundle[T]{
//		bundle: bundle[T, *cell.Output[T]]{
//			cells:        make([]*cell.Output[T], 0),
//			number:      number,
//			numberFloat: T(float64(number)),
//		},
//	}
//
//	if len(data) > 0 {
//		for _, v := range data {
//			target := v
//			bundle.cells = append(bundle.cells, cell.NewOutput(&target))
//		}
//	}
//
//	return bundle
//}

func (b *bundle[T, _]) SetTargets(data *[]T) {
	if len(*data) > len(b.cells) {
		utils.Logger.Error("data length is greater than bundle length")
	}
	if o, ok := any(b).(*cell.Output[T]); ok {
		for _, v := range *data {
			o.SetTarget(&v)
		}
	}
}

// Hidden bundle specific methods

//func NewHiddenBundle[T utils.Float](data []T) HiddenBundle[T] {
//	number := uint(len(data))
//	bundle := HiddenBundle[T]{
//		bundle: bundle[T, *cell.Hidden[T]]{
//			cells:        make([]*cell.Hidden[T], 0),
//			number:      number,
//			numberFloat: T(float64(number)),
//		},
//	}
//
//	if len(data) > 0 {
//		for range data {
//			bundle.cells = append(bundle.cells, cell.NewHidden[T]())
//		}
//	}
//
//	return bundle
//}

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

// bundle for Output or Hidden

func (b *bundle[T, _]) GetMisses() *[]*T {
	//misses := make([]*T, b.number)
	if c, ok := any(b.cells).([]neuron.Neuron[T]); ok {
		for i, n := range c {
			//misses[i] = n.GetMiss()
			b.cache[i] = n.GetMiss()
		}
	}
	return &b.cache
}

func (b *bundle[T, _]) calculateValues() {
	if c, ok := any(b.cells).([]neuron.Neuron[T]); ok {
		for _, n := range c {
			n.CalculateValue()
		}
	}
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

// bundle for Output or Hidden

func (b *bundle[T, _]) calculateWeights(rate *T) {
	if c, ok := any(b.cells).([]neuron.Neuron[T]); ok {
		for _, n := range c {
			n.CalculateWeight(rate)
		}
	}
}
