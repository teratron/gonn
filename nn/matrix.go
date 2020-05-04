package nn

import (
	_ "math/rand"
	_ "time"
)

const (
	DEFRATE float32 = .3      // Default rate
	MINLOSS float32 = .001    // The minimum value of the sum of the average square error at which the training is forcibly terminated
	MAXITER int     = 1000000 // The maximum number of iterations after which training is forcibly terminated
)

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
	X      int      // Индекс слоя в матрице
	Size   int      // Number of neurons in the layer
	Neuron []Neuron //
	Error  []Error  //
}

type Neuron struct {
	X, Y       int     // X - индекс слоя в матрице, Y - индекс нейрона в слое
	Value      float32 // Neuron value
	Activation         //
}

type Error struct {
	X, Y  int     //
	Size  int     //
	Value float32 // Neuron value
}

// Collection of weight parameters
type Synapse struct {
	Size   []int // Number of weight relationships {X, Y}, X-input (previous) layer, Y-output (next) layer
	Weight       // Weight value
}

type (
	//Neuron []float32
	//Error  []float32
	Weight [][]float32
)

//+-------------------------------------------------------------+
//|	Synapse														|
//+-------------------------------------------------------------+
// The function fills all weights with random numbers from -0.5 to 0.5
/*func (s *Synapse) Set() {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < s.Index; i++ {
		n := s.Synapse[i].Size[0] - 1
		for j := 0; j < s.Synapse[i].Size[0]; j++ {
			for k := 0; k < s.Synapse[i].Size[1]; k++ {
				if j == n && s.Bias == 0 {
					s.Synapse[i].Weight[j][k] = 0
				} else {
					s.Synapse[i].Weight[j][k] = rand.Float32() - .5
				}
			}
		}
	}
}*/

// Weights update function
func (s *Synapse) Get() float32 {
	return 0
}

//+-------------------------------------------------------------+
//|	Weight														|
//+-------------------------------------------------------------+
//
func (w *Weight) Set() {
}

// Weights update function
func (w *Weight) Get() float32 {
	return 0
}

//+-------------------------------------------------------------+
//|	Layer														|
//+-------------------------------------------------------------+
//
/*func (l *Layer) Get() float32 {
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j := 0; j < m.Layer[i].Size; j++ {
			var sum float32 = 0
			for k, v := range m.Layer[n].Neuron {
				sum += v * m.Synapse[n].Weight[k][j]
			}
			m.Layer[i].Neuron[j] = GetActivation(sum, m.Mode)
		}
	}
	return 0
}*/

//
/*func (l *Layer) Set(Setter)  {
}*/

//+-------------------------------------------------------------+
//|	Neuron														|
//+-------------------------------------------------------------+
// Function for calculating the values of neurons in a layer
/*func (n *Neuron) Get() float32 {
	var sum float32 = 0
	x := n.X - 1
	for k, v := range m.Layer[x].Neuron {
		sum += v * m.Synapse[x].Weight[k][n.Y]
	}
	n.Value = n.Activation.Get(sum)
	return n.Value
}*/

//
/*func (n *Neuron) Set() {
}*/

//+-------------------------------------------------------------+
//|	Error														|
//+-------------------------------------------------------------+
//
/*func (e *Error) Set() {
}*/

//
func (e *Error) Get() float32 {
	return 0
}
