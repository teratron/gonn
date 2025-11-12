package nn

/*import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
)

// Input устанавливает размер входного слоя
func (n *NN[T]) Input(size uint) *NN[T] {
	n.inputSize = size
	n.inputSet = true
	return n
}

// Dense добавляет полносвязный скрытый слой
func (n *NN[T]) Dense(size uint, activation activation.Type, bias bool) *NN[T] {
	n.hiddenLayers = append(n.hiddenLayers, HiddenLayer{
		Number:     size,
		Activation: activation,
		Bias:       bias, //n.useBias,
	})
	return n
}

// Hidden - альтернативное название для Dense
func (n *NN[T]) Hidden(size uint, activation activation.Type, bias bool) *NN[T] {
	return n.Dense(size, activation, bias)
}

// Output устанавливает выходной слой
func (n *NN[T]) Output(size uint, activation activation.Type) *NN[T] {
	n.outputSize = size
	n.outputActivation = activation
	n.outputSet = true
	return n
}

// WithLoss устанавливает функцию потерь
func (n *NN[T]) WithLoss(lossType loss.Type) *NN[T] {
	n.lossFunction = lossType
	return n
}

// WithLearningRate устанавливает скорость обучения
func (n *NN[T]) WithLearningRate(rate T) *NN[T] {
	n.learningRate = rate
	return n
}

// WithBias включает/выключает bias для всех слоев
func (n *NN[T]) WithBias(use bool) *NN[T] {
	n.useBias = use
	// Применяем ко всем уже добавленным слоям
	for i := range n.hiddenLayers {
		n.hiddenLayers[i].Bias = use
	}
	return n
}

// WithWeightInit устанавливает метод инициализации весов
func (n *NN[T]) WithWeightInit(method string) *NN[T] {
	n.weightInit = method
	return n
}

// Compile завершает конфигурацию и создает сеть
func (n *NN[T]) Compile() (*NN[T], error) {
	// Валидация
	if !n.inputSet {
		return nil, utils.NewError("input layer not configured")
	}
	if !n.outputSet {
		return nil, utils.NewError("output layer not configured")
	}
	if n.learningRate == 0 {
		n.learningRate = T(0.01) // значение по умолчанию
	}
	if n.lossFunction == "" {
		n.lossFunction = loss.MSE // значение по умолчанию
	}

	// Создаем сеть
	nn := &NN[T]{
		LearningRate: n.learningRate,
		Loss:         n.lossFunction,
	}

	// Инициализируем входной слой
	nn.Network.SetInputLayer(n.inputSize)

	// Инициализируем скрытые слои
	if len(n.hiddenLayers) > 0 {
		nn.Network.SetHiddenLayers(n.hiddenLayers...)
	}

	// Инициализируем выходной слой
	nn.Network.SetOutputLayer(n.outputSize, n.outputActivation, n.useBias)

	// Строим связи между слоями
	if err := nn.Network.Build(n.weightInit); err != nil {
		return nil, err
	}

	return nn, nil
}
*/
