package nn

import (
	"encoding/json"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// LayerConfig конфигурация одного слоя
type LayerConfig struct {
	Size       int
	Activation activation.Type
	UseBias    bool
	Name       string // опционально для отладки
}

// NetworkConfig конфигурация всей сети
type NetworkConfig[T utils.Float] struct {
	InputSize        int
	Layers           []LayerConfig
	OutputSize       int
	OutputActivation activation.Type
	LossFunction     loss.Type
	LearningRate     T

	// Дополнительные параметры
	WeightInit string
	Optimizer  string // "sgd", "adam", "rmsprop"
	Momentum   T

	// Регуляризация
	L1      T
	L2      T
	Dropout T
}

// ConfigBuilder строит NetworkConfig с fluent API
type ConfigBuilder[T utils.Float] struct {
	config NetworkConfig[T]
}

// NewConfig создает новый builder конфигурации
func NewConfig[T utils.Float]() *ConfigBuilder[T] {
	return &ConfigBuilder[T]{
		config: NetworkConfig[T]{
			LearningRate: T(0.01),
			LossFunction: loss.MSE,
			WeightInit:   "xavier",
			Optimizer:    "sgd",
		},
	}
}

// Input задает размер входа
func (c *ConfigBuilder[T]) Input(size int) *ConfigBuilder[T] {
	c.config.InputSize = size
	return c
}

// AddLayer добавляет слой с полной конфигурацией
func (c *ConfigBuilder[T]) AddLayer(size int, activation activation.Type) *ConfigBuilder[T] {
	c.config.Layers = append(c.config.Layers, LayerConfig{
		Size:       size,
		Activation: activation,
		UseBias:    true,
	})
	return c
}

// AddLayerWithName добавляет именованный слой
func (c *ConfigBuilder[T]) AddLayerWithName(size int, activation activation.Type, name string) *ConfigBuilder[T] {
	c.config.Layers = append(c.config.Layers, LayerConfig{
		Size:       size,
		Activation: activation,
		UseBias:    true,
		Name:       name,
	})
	return c
}

// Output задает выходной слой
func (c *ConfigBuilder[T]) Output(size int, activation activation.Type) *ConfigBuilder[T] {
	c.config.OutputSize = size
	c.config.OutputActivation = activation
	return c
}

// Loss задает функцию потерь
func (c *ConfigBuilder[T]) Loss(lossType loss.Type) *ConfigBuilder[T] {
	c.config.LossFunction = lossType
	return c
}

// LearningRate задает скорость обучения
func (c *ConfigBuilder[T]) LearningRate(rate T) *ConfigBuilder[T] {
	c.config.LearningRate = rate
	return c
}

// WeightInitialization задает метод инициализации весов
func (c *ConfigBuilder[T]) WeightInitialization(method string) *ConfigBuilder[T] {
	c.config.WeightInit = method
	return c
}

// Optimizer задает оптимизатор
func (c *ConfigBuilder[T]) Optimizer(opt string) *ConfigBuilder[T] {
	c.config.Optimizer = opt
	return c
}

// WithMomentum добавляет momentum
func (c *ConfigBuilder[T]) WithMomentum(m T) *ConfigBuilder[T] {
	c.config.Momentum = m
	return c
}

// L1Regularization добавляет L1 регуляризацию
func (c *ConfigBuilder[T]) L1Regularization(lambda T) *ConfigBuilder[T] {
	c.config.L1 = lambda
	return c
}

// L2Regularization добавляет L2 регуляризацию
func (c *ConfigBuilder[T]) L2Regularization(lambda T) *ConfigBuilder[T] {
	c.config.L2 = lambda
	return c
}

// Dropout добавляет dropout
func (c *ConfigBuilder[T]) Dropout(rate T) *ConfigBuilder[T] {
	c.config.Dropout = rate
	return c
}

// Build возвращает готовую конфигурацию
func (c *ConfigBuilder[T]) Build() NetworkConfig[T] {
	return c.config
}

// BuildAndCreate создает сеть из конфигурации
func (c *ConfigBuilder[T]) BuildAndCreate() (*NN[T], error) {
	config := c.Build()
	return CreateFromConfig(config)
}

// CreateFromConfig создает сеть из готовой конфигурации
func CreateFromConfig[T utils.Float](config NetworkConfig[T]) (*NN[T], error) {
	if config.InputSize <= 0 {
		return nil, utils.NewError("input size must be positive")
	}
	if config.OutputSize <= 0 {
		return nil, utils.NewError("output size must be positive")
	}

	nn := &NN[T]{
		LearningRate: config.LearningRate,
		Loss:         config.LossFunction,
	}

	// Создаем входной слой
	nn.Network.SetInputLayer(config.InputSize)

	// Создаем скрытые слои
	hiddenLayers := make([]HiddenLayer, len(config.Layers))
	for i, layer := range config.Layers {
		hiddenLayers[i] = HiddenLayer{
			Number:     layer.Size,
			Activation: layer.Activation,
			Bias:       layer.UseBias,
		}
	}
	if len(hiddenLayers) > 0 {
		nn.Network.SetHiddenLayers(hiddenLayers...)
	}

	// Создаем выходной слой
	nn.Network.SetOutputLayer(config.OutputSize, config.OutputActivation, true)

	// Строим связи
	if err := nn.Network.Build(config.WeightInit); err != nil {
		return nil, err
	}

	return nn, nil
}

// ========================
// ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
// ========================

// Пример 1: Создание через конфигурацию
func ExampleWithConfig() *NN[float32] {
	config := NewConfig[float32]().
		Input(2).
		AddLayer(4, activation.SIGMOID).
		Output(1, activation.SIGMOID).
		LearningRate(0.3).
		Loss(loss.MSE).
		Build()

	nn, _ := CreateFromConfig(config)
	return nn
}

// Пример 2: Именованные слои для отладки
func ExampleNamedLayers() *NN[float32] {
	nn, _ := NewConfig[float32]().
		Input(784).
		AddLayerWithName(128, activation.ReLU, "hidden1").
		AddLayerWithName(64, activation.ReLU, "hidden2").
		AddLayerWithName(32, activation.ReLU, "hidden3").
		Output(10, activation.SOFTMAX).
		LearningRate(0.001).
		Loss(loss.CROSS_ENTROPY).
		BuildAndCreate()

	return nn
}

// Пример 3: С регуляризацией и оптимизацией
func ExampleAdvancedConfig() *NN[float64] {
	nn, _ := NewConfig[float64]().
		Input(100).
		AddLayer(64, activation.ReLU).
		AddLayer(32, activation.ReLU).
		Output(10, activation.SOFTMAX).
		LearningRate(0.001).
		Loss(loss.CROSS_ENTROPY).
		WeightInitialization("he").
		Optimizer("adam").
		WithMomentum(0.9).
		L2Regularization(0.01).
		Dropout(0.5).
		BuildAndCreate()

	return nn
}

// Пример 4: Переиспользование конфигурации
func ExampleReuseConfig() {
	// Создаем базовую конфигурацию
	baseConfig := NewConfig[float32]().
		Input(10).
		AddLayer(20, activation.ReLU).
		Output(5, activation.SOFTMAX).
		LearningRate(0.01).
		Build()

	// Создаем несколько сетей с одинаковой архитектурой
	nn1, _ := CreateFromConfig(baseConfig)
	nn2, _ := CreateFromConfig(baseConfig)

	// Можно модифицировать конфигурацию
	modifiedConfig := baseConfig
	modifiedConfig.LearningRate = 0.001
	nn3, _ := CreateFromConfig(modifiedConfig)

	_, _, _ = nn1, nn2, nn3
}

// Пример 5: Валидация конфигурации
func (c NetworkConfig[T]) Validate() error {
	if c.InputSize <= 0 {
		return utils.NewError("input size must be positive")
	}
	if c.OutputSize <= 0 {
		return utils.NewError("output size must be positive")
	}
	if c.LearningRate <= 0 {
		return utils.NewError("learning rate must be positive")
	}
	for i, layer := range c.Layers {
		if layer.Size <= 0 {
			return utils.NewError("layer %d size must be positive", i)
		}
	}
	return nil
}

// Пример 6: Сохранение/загрузка конфигурации (JSON)
//import "encoding/json"

func (c NetworkConfig[T]) ToJSON() ([]byte, error) {
	return json.MarshalIndent(c, "", "  ")
}

func LoadConfigFromJSON[T utils.Float](data []byte) (NetworkConfig[T], error) {
	var config NetworkConfig[T]
	err := json.Unmarshal(data, &config)
	return config, err
}
