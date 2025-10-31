package cell

import (
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

// MockNeuron реализует интерфейс Neuron для тестирования
type MockNeuron[T utils.Float] struct {
	value T
	miss  T
}

func (m *MockNeuron[T]) GetValue() T {
	return m.value
}

func (m *MockNeuron[T]) GetMiss() T {
	return m.miss
}

func (m *MockNeuron[T]) CalculateValue() T {
	return m.value
}

func (m *MockNeuron[T]) CalculateWeight(miss T) T {
	return miss * m.value
}

func (m *MockNeuron[T]) Forward() T {
	return m.value
}

func (m *MockNeuron[T]) Backward(target T) T {
	m.miss = target - m.value
	return m.CalculateWeight(m.miss)
}

func TestAxonCreation(t *testing.T) {
	// Создаем мок-нейрон
	neuron := &MockNeuron[float32]{value: 0.5, miss: 0.1}
	
	// Создаем связь
	axon := NewAxon[float32](neuron, 0.8)
	
	// Проверяем инициализацию
	if axon.Target != neuron {
		t.Errorf("Expected target to be set correctly")
	}
	if axon.Weight != 0.8 {
		t.Errorf("Expected weight to be 0.8, got %v", axon.Weight)
	}
	if axon.Delta != 0 {
		t.Errorf("Expected delta to be 0, got %v", axon.Delta)
	}
}

func TestAxonWeightUpdate(t *testing.T) {
	// Создаем связь
	neuron := &MockNeuron[float32]{value: 0.5, miss: 0.1}
	axon := NewAxon[float32](neuron, 0.8)
	
	// Устанавливаем градиент
	axon.Delta = 0.05
	learningRate := float32(0.1)
	
	// Обновляем вес
	axon.UpdateWeight(learningRate)
	
	// Проверяем обновление
	expectedWeight := float32(0.8 - 0.05*0.1) // 0.795
	if axon.Weight != expectedWeight {
		t.Errorf("Expected weight to be %v, got %v", expectedWeight, axon.Weight)
	}
	if axon.Delta != 0 {
		t.Errorf("Expected delta to be reset to 0, got %v", axon.Delta)
	}
}

func TestActivationModes(t *testing.T) {
	tests := []struct {
		name     string
		mode     ActivationMode
		input    float32
		expected float32
	}{
		{"SIGMOID_0", SIGMOID, 0, 0.5},
		{"SIGMOID_positive", SIGMOID, 1, 0.7310586},
		{"SIGMOID_negative", SIGMOID, -1, 0.2689414},
		{"RELU_positive", RELU, 1, 1},
		{"RELU_negative", RELU, -1, 0},
		{"RELU_zero", RELU, 0, 0},
		{"TANH_0", TANH, 0, 0},
		{"TANH_positive", TANH, 1, 0.7615942},
		{"TANH_negative", TANH, -1, -0.7615942},
		{"LINEAR", LINEAR, 5, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			core := NewCoreCell[float32](tt.mode)
			result := core.applyActivation(tt.input)
			
			// Проверяем с некоторой точностью
			if abs(result-tt.expected) > 0.001 {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

// Вспомогательная функция для вычисления абсолютного значения
func abs[T utils.Float](x T) T {
	if x < 0 {
		return -x
	}
	return x
}

// TestNeuronInterface - временно отключён, так как BiasCell не реализует полный интерфейс Neuron
// В соответствии с Rust реализацией BiasCell реализует только Nucleus<T>, а не Neuron<T>
func TestNeuronInterface(t *testing.T) {
	// Создаем различные типы клеток для проверки интерфейса (кроме BiasCell)
	cells := []Neuron[float32]{
		//NewBiasCell[float32](), // BiasCell не реализует полный интерфейс Neuron в Rust версии
		NewInputCell[float32](nil),
		NewHiddenCell[float32](SIGMOID),
		NewOutputCell[float32](SIGMOID),
	}

	for i, cell := range cells {
		t.Run("NeuronInterface", func(t *testing.T) {
			// Проверяем, что все клетки реализуют интерфейс Neuron
			if cell == nil {
				t.Errorf("Cell %d is nil", i)
			}
			
			// Проверяем базовые методы
			value := cell.GetValue()
			if value < -1000 || value > 1000 { // Разумные пределы
				t.Errorf("Value %v is out of reasonable range", value)
			}
			
			// Проверяем Forward
			forwardResult := cell.Forward()
			if forwardResult < -1000 || forwardResult > 1000 {
				t.Errorf("Forward result %v is out of reasonable range", forwardResult)
			}
			
			// Проверяем Backward
			backwardResult := cell.Backward(0.5)
			if backwardResult < -1000 || backwardResult > 1000 {
				t.Errorf("Backward result %v is out of reasonable range", backwardResult)
			}
		})
	}
}