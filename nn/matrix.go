package nn

type NN interface {
	Set(Setter)
	Get() Getter
}

type Setter interface {
}

type Getter interface {
}

type Checker interface {
	Check()
}

type Processor interface {
	Init()
	Train()
	Query()
	Test()
}

// Collection of neural network matrix parameters
type Matrix struct {
	Init    bool      // Matrix initialization flag
	Size    int       // Количество слоёв в нейросети (Input + Hidden + Output)
	Index   int       // Индекс выходного (последнего) слоя нейросети
	Mode    uint8     // Идентификатор функции активации
	Bias    float32   // Нейрон смещения: от 0 до 1
	Rate    float32   // Коэффициент обучения, от 0 до 1
	Limit   float32   // Минимальный (достаточный) уровень средней квадратичной суммы ошибки при обучения
	Hidden  []int     // Массив количеств нейронов в каждом скрытом слое
	Layer   []Layer   // Коллекция слоя
	Synapse []Synapse // Коллекция весов связей
}

// Collection of neural layer parameters
type Layer struct {
	Size   int // Number of neurons in the layer
	Neuron     // Neuron value
	Error      // Error value
}

// Collection of weight parameters
type Synapse struct {
	Size   []int // Number of weight relationships {X, Y}, X-input (previous) layer, Y-output (next) layer
	Weight       // Weight value
}

type (
	Neuron []float32
	Error  []float32
	Weight [][]float32
)
