package axon

import (
	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// Axon структура определена в axon.go, но нам нужно определить дополнительные типы
// которые используются в других пакетах

// AxonInterface определяет интерфейс для аксона
type AxonInterface[T utils.Float] interface {
	CalculateValue() T
	CalculateMiss() T
	CalculateWeight(gradient T) 
}

// TargetInterface определяет интерфейс для цели (выходного нейрона)
type TargetInterface[T utils.Float] interface {
	pkg.Neuron[T]
	Forward() T
	UpdateWeight(learningRate T)
}

// DeltaInterface определяет интерфейс для дельты (разности)
type DeltaInterface[T utils.Float] interface {
	GetDelta() T
	SetDelta(delta T)
}