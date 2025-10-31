package cell

import (
	"github.com/teratron/gonn/pkg/utils"
)

// HiddenCell представляет скрытый слой нейронной сети
// Содержит CoreCell и исходящие связи (outgoing_axons)
// Аналог Rust HiddenCell
type HiddenCell[T utils.Float] struct {
	Core         *CoreCell[T]
	OutgoingAxons []*Axon[T]
}

// NewHiddenCell создает новую скрытую клетку
func NewHiddenCell[T utils.Float](activationMode ActivationMode) *HiddenCell[T] {
	return &HiddenCell[T]{
		Core:         NewCoreCell[T](activationMode),
		OutgoingAxons: make([]*Axon[T], 0),
	}
}

// GetValue возвращает значение скрытой клетки
func (h *HiddenCell[T]) GetValue() T {
	return h.Core.GetValue()
}

// GetMiss возвращает ошибку скрытой клетки
func (h *HiddenCell[T]) GetMiss() T {
	return h.Core.Miss
}

// CalculateValue вычисляет значение скрытой клетки
func (h *HiddenCell[T]) CalculateValue() T {
	return h.Core.CalculateValue()
}

// CalculateWeight вычисляет вес скрытой клетки
func (h *HiddenCell[T]) CalculateWeight(miss T) T {
	return h.Core.CalculateWeight(miss)
}

// Forward выполняет прямое распространение для скрытой клетки
func (h *HiddenCell[T]) Forward() T {
	value := h.CalculateValue()
	h.Core.Value = value
	return value
}

// Backward выполняет обратное распространение ошибки
func (h *HiddenCell[T]) Backward(target T) T {
	// Для скрытых слоев target не используется напрямую
	// Ошибка вычисляется на основе градиентов от следующего слоя
	h.Core.Miss = target
	return h.CalculateWeight(target)
}

// AddOutgoingConnection добавляет исходящую связь
func (h *HiddenCell[T]) AddOutgoingConnection(target Neuron[T], weight T) {
	axon := NewAxon[T](target, weight)
	h.OutgoingAxons = append(h.OutgoingAxons, axon)
}

// PropagateForward распространяет сигнал вперед по всем исходящим связям
func (h *HiddenCell[T]) PropagateForward() {
	h.Forward()
	for _, axon := range h.OutgoingAxons {
		axon.Target.Forward()
		// Здесь можно добавить логику накопления входных сигналов
	}
}

// PropagateBackward распространяет ошибку назад по всем исходящим связям
func (h *HiddenCell[T]) PropagateBackward(learningRate T) {
	for _, axon := range h.OutgoingAxons {
		// Вычисляем градиент для связи
		axon.Delta = h.GetMiss() * axon.Target.GetValue()
		axon.UpdateWeight(learningRate)
	}
}