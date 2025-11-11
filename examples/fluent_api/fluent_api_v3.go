package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// Option функциональная опция для конфигурации сети
type Option[T utils.Float] func(*networkOptions[T])

// networkOptions внутренняя структура для хранения опций
type networkOptions[T utils.Float] struct {
	inputSize        int
	layers           []layerOption[T]
	outputSize       int
	outputActivation activation.Type
	learningRate     T
	lossFunction     loss.Type
	useBias          bool
	weightInit       string

	// Callbacks
	onEpochEnd func(epoch int, loss T)
	onBatchEnd func(batch int, loss T)
}

type layerOption[T utils.Float] struct {
	size       int
	activation activation.Type
	useBias    bool
	dropout    T
}

// WithInput устанавливает входной слой
func WithInput[T utils.Float](size int) Option[T] {
	return func(o *networkOptions[T]) {
		o.inputSize = size
	}
}

// WithHiddenLayer добавляет скрытый слой
func WithHiddenLayer[T utils.Float](size int, activation activation.Type) Option[T] {
	return func(o *networkOptions[T]) {
		o.layers = append(o.layers, layerOption[T]{
			size:       size,
			activation: activation,
			useBias:    o.useBias,
		})
	}
}

// WithOutput устанавливает выходной слой
func WithOutput[T utils.Float](size int, activation activation.Type) Option[T] {
	return func(o *networkOptions[T]) {
		o.outputSize = size
		o.outputActivation = activation
	}
}

// WithLearningRate устанавливает скорость обучения
func WithLearningRate[T utils.Float](rate T) Option[T] {
	return func(o *networkOptions[T]) {
		o.learningRate = rate
	}
}

// WithLoss устанавливает функцию потерь
func WithLoss[T utils.Float](lossType loss.Type) Option[T] {
	return func(o *networkOptions[T]) {
		o.lossFunction = lossType
	}
}

// WithBias включает/выключает bias
func WithBias[T utils.Float](use bool) Option[T] {
	return func(o *networkOptions[T]) {
		o.useBias = use
	}
}

// WithWeightInit устанавливает метод инициализации весов
func WithWeightInit[T utils.Float](method string) Option[T] {
	return func(o *networkOptions[T]) {
		o.weightInit = method
	}
}

// WithEpochCallback устанавливает callback на конец эпохи
func WithEpochCallback[T utils.Float](callback func(epoch int, loss T)) Option[T] {
	return func(o *networkOptions[T]) {
		o.onEpochEnd = callback
	}
}

// WithBatchCallback устанавливает callback на конец батча
func WithBatchCallback[T utils.Float](callback func(batch int, loss T)) Option[T] {
	return func(o *networkOptions[T]) {
		o.onBatchEnd = callback
	}
}

// NewNetwork создает сеть с функциональными опциями
func NewNetwork[T utils.Float](opts ...Option[T]) (*NN[T], error) {
	// Дефолтные значения
	options := &networkOptions[T]{
		learningRate: T(0.01),
		lossFunction: loss.MSE,
		useBias:      true,
		weightInit:   "xavier",
	}

	// Применяем опции
	for _, opt := range opts {
		opt(options)
	}

	// Валидация
	if options.inputSize <= 0 {
		return nil, utils.NewError("input size not set or invalid")
	}
	if options.outputSize <= 0 {
		return nil, utils.NewError("output size not set or invalid")
	}

	// Создаем сеть
	nn := &NN[T]{
		LearningRate: options.learningRate,
		Loss:         options.lossFunction,
	}

	// Устанавливаем входной слой
	nn.Network.SetInputLayer(options.inputSize)

	// Устанавливаем скрытые слои
	if len(options.layers) > 0 {
		hiddenLayers := make([]HiddenLayer, len(options.layers))
		for i, layer := range options.layers {
			hiddenLayers[i] = HiddenLayer{
				Number:     layer.size,
				Activation: layer.activation,
				Bias:       layer.useBias,
			}
		}
		nn.Network.SetHiddenLayers(hiddenLayers...)
	}

	// Устанавливаем выходной слой
	nn.Network.SetOutputLayer(options.outputSize, options.outputActivation, options.useBias)

	// Строим связи
	if err := nn.Network.Build(options.weightInit); err != nil {
		return nil, err
	}

	return nn, nil
}

// MustNewNetwork как NewNetwork, но паникует при ошибке
func MustNewNetwork[T utils.Float](opts ...Option[T]) *NN[T] {
	nn, err := NewNetwork(opts...)
	if err != nil {
		panic(err)
	}
	return nn
}

// ========================
// КОМБИНИРОВАННЫЕ ОПЦИИ (Higher-Order Options)
// ========================

// Sequential создает последовательность слоев одинакового размера
func Sequential[T utils.Float](count int, size int, activation activation.Type) Option[T] {
	return func(o *networkOptions[T]) {
		for i := 0; i < count; i++ {
			WithHiddenLayer[T](size, activation)(o)
		}
	}
}

// DeepNetwork создает глубокую сеть с уменьшающимся размером слоев
func DeepNetwork[T utils.Float](startSize int, layers int, activation activation.Type) Option[T] {
	return func(o *networkOptions[T]) {
		size := startSize
		for i := 0; i < layers; i++ {
			WithHiddenLayer[T](size, activation)(o)
			size = size / 2
			if size < 2 {
				size = 2
			}
		}
	}
}

// StandardSetup стандартная конфигурация для большинства задач
func StandardSetup[T utils.Float](learningRate T) Option[T] {
	return func(o *networkOptions[T]) {
		WithLearningRate[T](learningRate)(o)
		WithLoss[T](loss.MSE)(o)
		WithBias[T](true)(o)
		WithWeightInit[T]("xavier")(o)
	}
}

// ========================
// ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
// ========================

// Пример 1: Простая сеть XOR
func ExampleSimpleXOR() *NN[float32] {
	return MustNewNetwork[float32](
		WithInput[float32](2),
		WithHiddenLayer[float32](4, activation.SIGMOID),
		WithOutput[float32](1, activation.SIGMOID),
		WithLearningRate[float32](0.3),
		WithLoss[float32](loss.MSE),
	)
}

// Пример 2: Используя комбинированные опции
func ExampleDeepNetworkWithOptions() *NN[float32] {
	return MustNewNetwork[float32](
		WithInput[float32](784),
		DeepNetwork[float32](128, 3, activation.ReLU), // 3 слоя: 128, 64, 32
		WithOutput[float32](10, activation.SOFTMAX),
		StandardSetup[float32](0.001),
	)
}

// Пример 3: Сеть с callbacks
func ExampleWithCallbacks() *NN[float32] {
	return MustNewNetwork[float32](
		WithInput[float32](10),
		WithHiddenLayer[float32](20, activation.ReLU),
		WithHiddenLayer[float32](15, activation.ReLU),
		WithOutput[float32](5, activation.SOFTMAX),
		WithLearningRate[float32](0.01),
		WithEpochCallback[float32](func(epoch int, loss float32) {
			if epoch%100 == 0 {
				utils.Logger.Info("Training", "epoch", epoch, "loss", loss)
			}
		}),
	)
}

// Пример 4: Последовательная сеть
func ExampleSequential() *NN[float64] {
	return MustNewNetwork[float64](
		WithInput[float64](100),
		Sequential[float64](5, 50, activation.ReLU), // 5 слоев по 50 нейронов
		WithOutput[float64](10, activation.SOFTMAX),
		WithLearningRate[float64](0.001),
		WithLoss[float64](loss.CROSS_ENTROPY),
	)
}

// Пример 5: Создание с проверкой ошибок
func ExampleWithErrorHandling() (*NN[float32], error) {
	nn, err := NewNetwork[float32](
		WithInput[float32](5),
		WithHiddenLayer[float32](10, activation.TanH),
		WithOutput[float32](2, activation.SIGMOID),
		WithLearningRate[float32](0.05),
	)

	if err != nil {
		utils.Logger.Error("Failed to create network", "error", err)
		return nil, err
	}

	return nn, nil
}

// Пример 6: Без bias
func ExampleNoBias() *NN[float32] {
	return MustNewNetwork[float32](
		WithInput[float32](3),
		WithBias[float32](false), // Отключаем bias для всех слоев
		WithHiddenLayer[float32](5, activation.ReLU),
		WithHiddenLayer[float32](5, activation.ReLU),
		WithOutput[float32](1, activation.Linear),
		WithLearningRate[float32](0.01),
	)
}

// Пример 7: Создание нескольких сетей с общими опциями
func ExampleSharedOptions() []*NN[float32] {
	// Общие опции
	commonOpts := []Option[float32]{
		WithLearningRate[float32](0.001),
		WithLoss[float32](loss.MSE),
		WithWeightInit[float32]("he"),
	}

	// Сеть 1
	nn1 := MustNewNetwork[float32](append(commonOpts,
		WithInput[float32](10),
		WithHiddenLayer[float32](20, activation.ReLU),
		WithOutput[float32](5, activation.Linear),
	)...)

	// Сеть 2
	nn2 := MustNewNetwork[float32](append(commonOpts,
		WithInput[float32](15),
		WithHiddenLayer[float32](30, activation.ReLU),
		WithOutput[float32](8, activation.Linear),
	)...)

	return []*NN[float32]{nn1, nn2}
}

// ========================
// ПРЕСЕТЫ ДЛЯ ТИПОВЫХ ЗАДАЧ
// ========================

// PresetXOR создает сеть для решения XOR
func PresetXOR[T utils.Float]() *NN[T] {
	return MustNewNetwork[T](
		WithInput[T](2),
		WithHiddenLayer[T](4, activation.SIGMOID),
		WithOutput[T](1, activation.SIGMOID),
		WithLearningRate[T](0.3),
		WithLoss[T](loss.MSE),
	)
}

// PresetMNIST создает сеть для классификации MNIST
func PresetMNIST[T utils.Float]() *NN[T] {
	return MustNewNetwork[T](
		WithInput[T](784),
		WithHiddenLayer[T](128, activation.ReLU),
		WithHiddenLayer[T](64, activation.ReLU),
		WithOutput[T](10, activation.SOFTMAX),
		WithLearningRate[T](0.001),
		WithLoss[T](loss.CROSS_ENTROPY),
		WithWeightInit[T]("he"),
	)
}

// PresetRegression создает сеть для регрессии
func PresetRegression[T utils.Float](inputSize, hiddenSize int) *NN[T] {
	return MustNewNetwork[T](
		WithInput[T](inputSize),
		WithHiddenLayer[T](hiddenSize, activation.ReLU),
		WithHiddenLayer[T](hiddenSize/2, activation.ReLU),
		WithOutput[T](1, activation.Linear),
		WithLearningRate[T](0.01),
		WithLoss[T](loss.MSE),
	)
}
