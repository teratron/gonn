package network

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type Bundle[T utils.Float, S neuron.Nucleus[T]] struct {
	Cells  []S
	number uint
}

func NewBundle[T utils.Float, S neuron.Nucleus[T]](/*data *[]S*/) Bundle[T, S] {
	//number := uint(len(*data))
	return Bundle[T, S]{
		Cells:  make([]S, 0),
		number: 0,
	}
}

func (b *Bundle[T, S]) Add(cell S) {
	b.Cells = append(b.Cells, cell)
	b.number++
}

// Bundle for Output or Hidden.
func (b *Bundle[T, S]) GetMisses() []T {
	//self.cells.iter().map(|n| n.get_miss()).collect::<Vec<&T>>()
	for _, cell := range b.Cells {
		return cell.GetMisses()
	}
}

/*
pub(super) struct Bundle<T, S> {
    /// Reference to a slice of neurons.
    pub(super) cells: Box<Vec<S>>,

    /// Number neurons.
    pub(super) _number: usize,
    pub(super) _number_float: T,

    _marker: PhantomData<T>,
}

// Common Bundle.
impl<T, S> Bundle<T, S>
where
    T: Float,
{
    pub(super) fn new() -> Self {
        Self {
            cells: Box::new(Vec::new()),
            _number: 0,
            _number_float: T::ZERO,
            _marker: PhantomData,
        }
    }

    pub(super) fn new_with(data: &[T]) -> Self {
        let number = data.len();
        Self {
            cells: Box::new(Vec::new()),
            _number: number,
            _number_float: T::from_f64(number as f64),
            _marker: PhantomData,
        }
    }

    pub(super) fn set_number(&mut self, number: usize) {
        self.number = number;
        self.number_float = T::from(number as f64);
    }
}

// Bundle for Output or Hidden.
impl<T, S> Bundle<T, S>
where
    T: Float,
    S: Neuron<T>,
{
    pub(super) fn get_values(&self) -> Vec<&T> {
        self.cells
            .iter()
            .map(|n| n.get_value())
            .collect::<Vec<&T>>()
    }

    pub(super) fn get_misses(&self) -> Vec<&T> {
        self.cells.iter().map(|n| n.get_miss()).collect::<Vec<&T>>()
    }
}

// Input Bundle.
impl<'a, T: Float> Bundle<T, Input<'a, T>> {
    pub(super) fn _new(data: &'a [T]) -> Self {
        Self {
            cells: Box::new(if data.is_empty() {
                Vec::new()
            } else {
                data.iter().map(|v| Input::new(v)).collect()
            }),
            ..Self::new_with(data)
        }
    }

    /// Sets the input data for the network.
    pub(super) fn set_inputs(&mut self, data: &'a [T]) {
        data.iter()
            .enumerate()
            .for_each(|(i, v)| self.cells[i].set_value(v));
    }
}

// Output Bundle.
impl<'a, T: Float> Bundle<T, OutputCell<'a, T>> {
    pub(super) fn _new(data: &'a [T]) -> Self {
        Self {
            cells: Box::new(data.iter().map(|v| OutputCell::new(v)).collect()),
            ..Self::new_with(data)
        }
    }

    /// Sets the target data for the network.
    pub(super) fn set_targets(&mut self, data: &'a [T]) {
        data.iter()
            .enumerate()
            .for_each(|(i, v)| self.cells[i].set_target(v));
    }
}

// Hidden Bundle.
impl<T: Float> Bundle<T, Hidden<T>> {
    pub(super) fn _new(data: &[T]) -> Self {
        Self {
            cells: Box::new(data.iter().map(|_| Hidden::new()).collect()),
            ..Self::new_with(data)
        }
    }
}

impl<T: Float, S> Default for Bundle<T, S> {
    fn default() -> Self {
        Self {
            cells: Box::new(Vec::new()),
            _number: 0,
            _number_float: T::ZERO,
            _marker: PhantomData,
        }
    }
}
*/
