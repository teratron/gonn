package network

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

type bundle[T utils.Float, S neuron.Nucleus[T]] struct {
	cells     []S
	cache     []*T
	size      int
	sizeFloat T
}

func newBundle[T utils.Float, S neuron.Nucleus[T]]() bundle[T, S] {
	return bundle[T, S]{
		cells:     make([]S, 0),
		cache:     make([]*T, 0),
		size:      0,
		sizeFloat: T(0),
	}
}

//func newBundleWithData[T utils.Float, S neuron.Nucleus[T]](data []T) bundle[T, S] {
//	size := len(data)
//	return bundle[T, S]{
//		cells:       make([]S, 0),
//		size:      size,
//		sizeFloat: T(float64(size)),
//	}
//}

func (b *bundle[T, S]) Add(cell S) {
	b.cells = append(b.cells, cell)
	b.cache = append(b.cache, nil)
	b.size++
	b.sizeFloat = T(b.size)
}

func (b *bundle[T, S]) GetValues() *[]*T {
	for i, c := range b.cells {
		b.cache[i] = c.GetValue()
	}
	return &b.cache
}

// Input bundle specific methods

//	func NewInputBundle[T utils.Float](data []T) InputBundle[T] {
//		size := uint(len(data))
//		bundle := InputBundle[T]{
//			bundle: bundle[T, *cell.Input[T]]{
//				cells:        make([]*cell.Input[T], 0),
//				size:      size,
//				sizeFloat: T(float64(size)),
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
	if len(*data) > b.size {
		utils.Logger.Error("data length is greater than bundle length")
		return
	}
	if i, ok := any(b.cells[0]).(*cell.Input[T]); ok {
		for _, v := range *data {
			i.SetValue(&v)
		}
	}
}

// Output bundle specific methods

//func NewOutputBundle[T utils.Float](data []T) *cell.Output[T] {
//	size := uint(len(data))
//	bundle := OutputBundle[T]{
//		bundle: bundle[T, *cell.Output[T]]{
//			cells:        make([]*cell.Output[T], 0),
//			size:      size,
//			sizeFloat: T(float64(size)),
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
	if len(*data) > b.size {
		utils.Logger.Error("data length is greater than bundle length")
		return
	}
	if o, ok := any(b).(*cell.Output[T]); ok {
		for _, v := range *data {
			o.SetTarget(&v)
		}
	}
}

// Hidden bundle specific methods

//func NewHiddenBundle[T utils.Float](data []T) HiddenBundle[T] {
//	size := uint(len(data))
//	bundle := HiddenBundle[T]{
//		bundle: bundle[T, *cell.Hidden[T]]{
//			cells:        make([]*cell.Hidden[T], 0),
//			size:      size,
//			sizeFloat: T(float64(size)),
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
	if c, ok := any(b.cells).([]neuron.Neuron[T]); ok {
		for i, n := range c {
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
