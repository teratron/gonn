package axon

import (
	"math/rand"
	"time"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// AxonBundle представляет коллекцию аксонов
// Аналог Rust типа AxonBundle<T> = Vec<Axon<T>>
type AxonBundle[T utils.Float] []Axon[T]

// Axon представляет связь между клетками нейронной сети
// Аналог Rust структуры Axon<T>
type Axon[T utils.Float] struct {
	// Вес аксона
	Weight T

	// Входная клетка: HiddenCell, InputCell, BiasCell
	IncomingCell pkg.Nucleus[T]

	// Выходная клетка: HiddenCell, OutputCell
	OutgoingCell pkg.Neuron[T]
}

// New создает новый аксон со случайной инициализацией веса в диапазоне [-0.5, 0.5]
// Аналог Rust конструктора pub(super) fn new()
func New[T utils.Float](
	incomingCell pkg.Nucleus[T],
	outgoingCell pkg.Neuron[T],
) *Axon[T] {
	// Инициализация генератора случайных чисел с текущим временем
	// Create a local random generator with current time as seed
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	return &Axon[T]{
		Weight:       T(rng.Float64()*1.0 - 0.5), // Случайное значение в диапазоне [-0.5, 0.5]
		IncomingCell: incomingCell,
		OutgoingCell: outgoingCell,
	}
}

// CalculateValue выполняет прямое распространение сигнала
// Аналог Rust метода pub(super) fn calculate_value(&self) -> T
// Формула: *incoming_cell.GetValue() * weight
func (a *Axon[T]) CalculateValue() T {
	return a.IncomingCell.GetValue() * a.Weight
}

// CalculateMiss выполняет обратное распространение ошибки
// Аналог Rust метода pub(super) fn calculate_miss(&self) -> T
// Формула: *outgoing_cell.GetMiss() * weight
func (a *Axon[T]) CalculateMiss() T {
	return a.OutgoingCell.GetMiss() * a.Weight
}

// CalculateWeight обновляет вес аксона на основе градиента
// Аналог Rust метода pub(super) fn calculate_weight(&mut self, gradient: &T)
// Формула: weight += gradient * *incoming_cell.GetValue()
func (a *Axon[T]) CalculateWeight(gradient T) {
	a.Weight += gradient * a.IncomingCell.GetValue()
}