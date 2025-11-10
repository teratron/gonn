package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// NetworkBuilder - строитель сети с fluent API
type NetworkBuilder[T utils.Float] struct {
	inputSize        int
	hiddenLayers     []HiddenLayer
	outputSize       int
	outputActivation activation.Type
	lossFunction     loss.Type
	learningRate     T

	// Опциональные параметры
	useBias    bool
	weightInit string // "xavier", "he", "random"

	// Флаги для валидации
	inputSet  bool
	outputSet bool
}

// New создает новый builder
func NewBuilder[T utils.Float]() *NetworkBuilder[T] {
	return &NetworkBuilder[T]{
		useBias:    true,
		weightInit: "xavier",
	}
}

// Input устанавливает размер входного слоя
func (b *NetworkBuilder[T]) Input(size int) *NetworkBuilder[T] {
	b.inputSize = size
	b.inputSet = true
	return b
}

// Dense добавляет полносвязный скрытый слой
func (b *NetworkBuilder[T]) Dense(neurons int, activation activation.Type) *NetworkBuilder[T] {
	b.hiddenLayers = append(b.hiddenLayers, HiddenLayer{
		Number:     neurons,
		Activation: activation,
		Bias:       b.useBias,
	})
	return b
}

// Hidden - альтернативное название для Dense
func (b *NetworkBuilder[T]) Hidden(neurons int, activation activation.Type) *NetworkBuilder[T] {
	return b.Dense(neurons, activation)
}

// Output устанавливает выходной слой
func (b *NetworkBuilder[T]) Output(size int, activation activation.Type) *NetworkBuilder[T] {
	b.outputSize = size
	b.outputActivation = activation
	b.outputSet = true
	return b
}

// WithLoss устанавливает функцию потерь
func (b *NetworkBuilder[T]) WithLoss(lossType loss.Type) *NetworkBuilder[T] {
	b.lossFunction = lossType
	return b
}

// WithLearningRate устанавливает скорость обучения
func (b *NetworkBuilder[T]) WithLearningRate(rate T) *NetworkBuilder[T] {
	b.learningRate = rate
	return b
}

// WithBias включает/выключает bias для всех слоев
func (b *NetworkBuilder[T]) WithBias(use bool) *NetworkBuilder[T] {
	b.useBias = use
	// Применяем ко всем уже добавленным слоям
	for i := range b.hiddenLayers {
		b.hiddenLayers[i].Bias = use
	}
	return b
}

// WithWeightInit устанавливает метод инициализации весов
func (b *NetworkBuilder[T]) WithWeightInit(method string) *NetworkBuilder[T] {
	b.weightInit = method
	return b
}

// Compile завершает конфигурацию и создает сеть
func (b *NetworkBuilder[T]) Compile() (*NN[T], error) {
	// Валидация
	if !b.inputSet {
		return nil, utils.NewError("input layer not configured")
	}
	if !b.outputSet {
		return nil, utils.NewError("output layer not configured")
	}
	if b.learningRate == 0 {
		b.learningRate = T(0.01) // значение по умолчанию
	}
	if b.lossFunction == "" {
		b.lossFunction = loss.MSE // значение по умолчанию
	}

	// Создаем сеть
	nn := &NN[T]{
		LearningRate: b.learningRate,
		Loss:         b.lossFunction,
	}

	// Инициализируем входной слой
	nn.Network.SetInputLayer(b.inputSize)

	// Инициализируем скрытые слои
	if len(b.hiddenLayers) > 0 {
		nn.Network.SetHiddenLayers(b.hiddenLayers...)
	}

	// Инициализируем выходной слой
	nn.Network.SetOutputLayer(b.outputSize, b.outputActivation, b.useBias)

	// Строим связи между слоями
	if err := nn.Network.Build(b.weightInit); err != nil {
		return nil, err
	}

	return nn, nil
}

// MustCompile как Compile, но паникует при ошибке (удобно для примеров)
func (b *NetworkBuilder[T]) MustCompile() *NN[T] {
	nn, err := b.Compile()
	if err != nil {
		panic(err)
	}
	return nn
}

// ========================
// ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
// ========================

// Пример 1: Простая сеть для XOR
func ExampleSimpleXOR() *NN[float32] {
	return NewBuilder[float32]().
		Input(2).
		Dense(4, activation.SIGMOID).
		Output(1, activation.SIGMOID).
		WithLearningRate(0.3).
		WithLoss(loss.MSE).
		MustCompile()
}

// Пример 2: Глубокая сеть с несколькими скрытыми слоями
func ExampleDeepNetwork() *NN[float32] {
	return NewBuilder[float32]().
		Input(784). // MNIST
		Dense(128, activation.RELU).
		Dense(64, activation.RELU).
		Dense(32, activation.RELU).
		Output(10, activation.SOFTMAX).
		WithLearningRate(0.001).
		WithLoss(loss.CROSS_ENTROPY).
		WithWeightInit("he").
		MustCompile()
}

// Пример 3: Сеть без bias
func ExampleNoBias() *NN[float64] {
	return NewBuilder[float64]().
		Input(10).
		WithBias(false).
		Dense(20, activation.TANH).
		Dense(15, activation.TANH).
		Output(5, activation.LINEAR).
		WithLearningRate(0.01).
		MustCompile()
}

// Пример 4: Регрессия
func ExampleRegression() *NN[float32] {
	return NewBuilder[float32]().
		Input(5).
		Dense(10, activation.RELU).
		Dense(10, activation.RELU).
		Output(1, activation.LINEAR). // Linear для регрессии
		WithLearningRate(0.01).
		WithLoss(loss.MSE).
		MustCompile()
}

// Пример 5: Бинарная классификация
func ExampleBinaryClassification() *NN[float32] {
	return NewBuilder[float32]().
		Input(20).
		Dense(16, activation.RELU).
		Dense(8, activation.RELU).
		Output(1, activation.SIGMOID). // Sigmoid для бинарной классификации
		WithLearningRate(0.001).
		WithLoss(loss.BINARY_CROSS_ENTROPY).
		MustCompile()
}

// Пример 6: Многоклассовая классификация
func ExampleMulticlassClassification() *NN[float64] {
	return NewBuilder[float64]().
		Input(100).
		Dense(64, activation.RELU).
		Dense(32, activation.RELU).
		Output(10, activation.SOFTMAX). // Softmax для мультиклассовой
		WithLearningRate(0.001).
		WithLoss(loss.CROSS_ENTROPY).
		MustCompile()
}
