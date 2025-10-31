package pkg

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Nucleus - базовый интерфейс для всех клеток нейронной сети
// Аналог Rust trait Nucleus с методом get_value()
type Nucleus[T utils.Float] interface {
	// GetValue возвращает текущее значение клетки
	GetValue() *T
}

// Neuron - интерфейс для нейронов с возможностью обучения
// Аналог Rust trait Neuron (наследуется от Nucleus)
// Содержит методы для прямого и обратного распространения
type Neuron[T utils.Float] interface {
	Nucleus[T]
	
	// GetMiss возвращает ошибку (разницу между целевым и полученным значением)
	GetMiss() *T
	
	// CalculateValue вычисляет значение нейрона на основе входных сигналов
	CalculateValue() *T
	
	// CalculateWeight вычисляет вес нейрона на основе ошибки
	CalculateWeight(*T) T
	
	// Forward выполняет прямое распространение сигнала
	//Forward() *T
	
	// Backward выполняет обратное распространение ошибки
	//Backward(target *T) *T
}
