package cell

import (
	"testing"
)

func TestCoreCellCreation(t *testing.T) {
	core := NewCoreCell[float32](SIGMOID)
	
	if core == nil {
		t.Errorf("CoreCell creation failed")
	}
	
	if core.GetValue() != 0 {
		t.Errorf("Expected initial value to be 0, got %v", core.GetValue())
	}
	
	if core.Miss != 0 {
		t.Errorf("Expected initial miss to be 0, got %v", core.Miss)
	}
	
	if core.ActivationMode != SIGMOID {
		t.Errorf("Expected activation mode to be SIGMOID, got %v", core.ActivationMode)
	}
}

func TestCoreCellGetValue(t *testing.T) {
	core := NewCoreCell[float64](RELU)
	
	core.Value = 2.5
	
	if core.GetValue() != 2.5 {
		t.Errorf("Expected value to be 2.5, got %v", core.GetValue())
	}
}

func TestCoreCellAddIncomingConnection(t *testing.T) {
	core := NewCoreCell[float32](SIGMOID)
	
	// Создаем мок-нейрон для тестирования
	neuron := &MockNeuron[float32]{value: 1.0, miss: 0.1}
	
	core.AddIncomingConnection(neuron, 0.5)
	
	if len(core.IncomingAxons) != 1 {
		t.Errorf("Expected 1 incoming axon, got %d", len(core.IncomingAxons))
	}
	
	axon := core.IncomingAxons[0]
	if axon.Target != neuron {
		t.Errorf("Expected axon target to be the neuron")
	}
	if axon.Weight != 0.5 {
		t.Errorf("Expected axon weight to be 0.5, got %v", axon.Weight)
	}
}

func TestCoreCellCalculateValue(t *testing.T) {
	core := NewCoreCell[float32](LINEAR)
	
	// Добавляем входящие связи
	neuron1 := &MockNeuron[float32]{value: 1.0, miss: 0}
	neuron2 := &MockNeuron[float32]{value: 2.0, miss: 0}
	
	core.AddIncomingConnection(neuron1, 0.5)
	core.AddIncomingConnection(neuron2, 0.3)
	core.SetBias(0.1)
	
	result := core.CalculateValue()
	
	// Ожидаемое значение: 0.1 + 1.0*0.5 + 2.0*0.3 = 0.1 + 0.5 + 0.6 = 1.2
	expected := float32(1.2)
	if result != expected {
		t.Errorf("Expected calculated value to be %v, got %v", expected, result)
	}
}

func TestCoreCellApplyActivation(t *testing.T) {
	tests := []struct {
		name     string
		mode     ActivationMode
		input    float32
		expected float32
	}{
		{"SIGMOID_0", SIGMOID, 0, 0.5},
		{"SIGMOID_positive", SIGMOID, 1, 0.7310586},
		{"RELU_positive", RELU, 1, 1},
		{"RELU_negative", RELU, -1, 0},
		{"TANH_0", TANH, 0, 0},
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

func TestCoreCellCalculateWeight(t *testing.T) {
	core := NewCoreCell[float64](SIGMOID)
	core.Value = 2.0
	
	result := core.CalculateWeight(0.5)
	
	// Ожидаемое значение: 0.5 * 2.0 = 1.0
	expected := float64(1.0)
	if result != expected {
		t.Errorf("Expected calculated weight to be %v, got %v", expected, result)
	}
}

func TestCoreCellSetBias(t *testing.T) {
	core := NewCoreCell[float32](SIGMOID)
	
	core.SetBias(0.5)
	
	if core.GetBias() != 0.5 {
		t.Errorf("Expected bias to be 0.5, got %v", core.GetBias())
	}
}

func TestCoreCellGetBias(t *testing.T) {
	core := NewCoreCell[float64](RELU)
	core.Bias = 1.5
	
	if core.GetBias() != 1.5 {
		t.Errorf("Expected bias to be 1.5, got %v", core.GetBias())
	}
}

func TestCoreCellReset(t *testing.T) {
	core := NewCoreCell[float32](SIGMOID)
	
	// Устанавливаем значения
	core.Value = 2.5
	core.Miss = 0.3
	core.Bias = 0.1
	
	// Сбрасываем
	core.Reset()
	
	if core.GetValue() != 0 {
		t.Errorf("Expected value to be reset to 0, got %v", core.GetValue())
	}
	if core.Miss != 0 {
		t.Errorf("Expected miss to be reset to 0, got %v", core.Miss)
	}
}

func TestCoreCellGetActivationDerivative(t *testing.T) {
	tests := []struct {
		name     string
		mode     ActivationMode
		value    float32
		expected float32
	}{
		{"SIGMOID_0.5", SIGMOID, 0.5, 0.25}, // 0.5 * (1 - 0.5) = 0.25
		{"SIGMOID_0", SIGMOID, 0, 0},        // 0 * (1 - 0) = 0
		{"RELU_positive", RELU, 1, 1},
		{"RELU_negative", RELU, -1, 0},
		{"TANH_0", TANH, 0, 1},              // 1 - 0^2 = 1
		{"TANH_0.5", TANH, 0.5, 0.75},       // 1 - 0.5^2 = 0.75
		{"LINEAR", LINEAR, 5, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			core := NewCoreCell[float32](tt.mode)
			core.Value = tt.value
			
			result := core.GetActivationDerivative()
			
			// Проверяем с некоторой точностью
			if abs(result-tt.expected) > 0.001 {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestCoreCellWithMultipleAxons(t *testing.T) {
	core := NewCoreCell[float32](SIGMOID)
	
	// Добавляем несколько связей
	for i := 0; i < 5; i++ {
		neuron := &MockNeuron[float32]{value: float32(i), miss: 0}
		core.AddIncomingConnection(neuron, float32(i)*0.1)
	}
	
	if len(core.IncomingAxons) != 5 {
		t.Errorf("Expected 5 incoming axons, got %d", len(core.IncomingAxons))
	}
	
	// Проверяем, что все связи добавлены корректно
	for i, axon := range core.IncomingAxons {
		expectedWeight := float32(i) * 0.1
		if axon.Weight != expectedWeight {
			t.Errorf("Expected axon %d weight to be %v, got %v", i, expectedWeight, axon.Weight)
		}
	}
}

func TestCoreCellEmptyIncomingAxons(t *testing.T) {
	core := NewCoreCell[float32](LINEAR)
	core.SetBias(2.0)
	
	result := core.CalculateValue()
	
	// Ожидаемое значение: только bias = 2.0
	expected := float32(2.0)
	if result != expected {
		t.Errorf("Expected calculated value to be %v, got %v", expected, result)
	}
}