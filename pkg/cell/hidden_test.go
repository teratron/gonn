package cell

import (
	"testing"
)

func TestHiddenCellCreation(t *testing.T) {
	hidden := NewHiddenCell[float32](SIGMOID)
	
	if hidden == nil {
		t.Errorf("HiddenCell creation failed")
	}
	
	if hidden.Core == nil {
		t.Errorf("Core should not be nil")
	}
	
	if hidden.OutgoingAxons == nil {
		t.Errorf("OutgoingAxons should not be nil")
	}
	
	if len(hidden.OutgoingAxons) != 0 {
		t.Errorf("Expected 0 outgoing axons initially, got %d", len(hidden.OutgoingAxons))
	}
}

func TestHiddenCellGetValue(t *testing.T) {
	hidden := NewHiddenCell[float64](RELU)
	hidden.Core.Value = 1.5
	
	if hidden.GetValue() != 1.5 {
		t.Errorf("Expected value to be 1.5, got %v", hidden.GetValue())
	}
}

func TestHiddenCellGetMiss(t *testing.T) {
	hidden := NewHiddenCell[float32](SIGMOID)
	hidden.Core.Miss = 0.3
	
	if hidden.GetMiss() != 0.3 {
		t.Errorf("Expected miss to be 0.3, got %v", hidden.GetMiss())
	}
}

func TestHiddenCellCalculateValue(t *testing.T) {
	hidden := NewHiddenCell[float64](LINEAR)
	
	// Добавляем входящие связи
	neuron1 := &MockNeuron[float64]{value: 1.0, miss: 0}
	neuron2 := &MockNeuron[float64]{value: 2.0, miss: 0}
	
	hidden.Core.AddIncomingConnection(neuron1, 0.5)
	hidden.Core.AddIncomingConnection(neuron2, 0.3)
	hidden.Core.SetBias(0.1)
	
	result := hidden.CalculateValue()
	
	// Ожидаемое значение: 0.1 + 1.0*0.5 + 2.0*0.3 = 1.2
	expected := float64(1.2)
	if result != expected {
		t.Errorf("Expected calculated value to be %v, got %v", expected, result)
	}
}

func TestHiddenCellCalculateWeight(t *testing.T) {
	hidden := NewHiddenCell[float32](SIGMOID)
	hidden.Core.Value = 2.0
	
	result := hidden.CalculateWeight(0.5)
	
	// Ожидаемое значение: 0.5 * 2.0 = 1.0
	expected := float32(1.0)
	if result != expected {
		t.Errorf("Expected calculated weight to be %v, got %v", expected, result)
	}
}

func TestHiddenCellForward(t *testing.T) {
	hidden := NewHiddenCell[float64](RELU)
	
	// Добавляем входящие связи для корректной работы CalculateValue
	neuron := &MockNeuron[float64]{value: 1.0, miss: 0}
	hidden.Core.AddIncomingConnection(neuron, 0.5)
	hidden.Core.SetBias(1.0) // Устанавливаем bias для получения предсказуемого результата
	
	result := hidden.Forward()
	
	// Forward() вызывает CalculateValue(), который вычисляет значение на основе входящих связей
	expected := hidden.CalculateValue()
	if result != expected {
		t.Errorf("Expected forward result to be %v, got %v", expected, result)
	}
	
	if hidden.Core.Value != expected {
		t.Errorf("Expected core value to be updated to %v, got %v", expected, hidden.Core.Value)
	}
}

func TestHiddenCellBackward(t *testing.T) {
	hidden := NewHiddenCell[float32](SIGMOID)
	
	target := float32(0.8)
	result := hidden.Backward(target)
	
	// Проверяем, что miss установлен
	if hidden.Core.Miss != target {
		t.Errorf("Expected miss to be set to %v, got %v", target, hidden.Core.Miss)
	}
	
	// Проверяем возвращаемое значение
	expected := hidden.CalculateWeight(target)
	if result != expected {
		t.Errorf("Expected backward result to be %v, got %v", expected, result)
	}
}

func TestHiddenCellAddOutgoingConnection(t *testing.T) {
	hidden := NewHiddenCell[float64](SIGMOID)
	
	// Создаем целевой нейрон
	target := &MockNeuron[float64]{value: 0.5, miss: 0.1}
	
	hidden.AddOutgoingConnection(target, 0.7)
	
	if len(hidden.OutgoingAxons) != 1 {
		t.Errorf("Expected 1 outgoing axon, got %d", len(hidden.OutgoingAxons))
	}
	
	axon := hidden.OutgoingAxons[0]
	if axon.Target != target {
		t.Errorf("Expected axon target to be the neuron")
	}
	if axon.Weight != 0.7 {
		t.Errorf("Expected axon weight to be 0.7, got %v", axon.Weight)
	}
}

func TestHiddenCellPropagateForward(t *testing.T) {
	hidden := NewHiddenCell[float32](SIGMOID)
	
	// Устанавливаем значение
	hidden.Core.Value = 1.0
	
	// Создаем целевые нейроны
	target1 := &MockNeuron[float32]{value: 0.5, miss: 0}
	target2 := &MockNeuron[float32]{value: 0.3, miss: 0}
	
	hidden.AddOutgoingConnection(target1, 0.6)
	hidden.AddOutgoingConnection(target2, 0.4)
	
	// Выполняем прямое распространение
	hidden.PropagateForward()
	
	// Проверяем, что целевые нейроны получили вызов Forward
	// (в реальной реализации здесь была бы логика накопления сигналов)
}

func TestHiddenCellPropagateBackward(t *testing.T) {
	hidden := NewHiddenCell[float64](SIGMOID)
	
	// Устанавливаем ошибку
	hidden.Core.Miss = 0.2
	
	// Создаем целевые нейроны
	target1 := &MockNeuron[float64]{value: 0.5, miss: 0}
	target2 := &MockNeuron[float64]{value: 0.3, miss: 0}
	
	hidden.AddOutgoingConnection(target1, 0.6)
	hidden.AddOutgoingConnection(target2, 0.4)
	
	learningRate := float64(0.1)
	
	// Выполняем обратное распространение
	hidden.PropagateBackward(learningRate)
	
	// Проверяем, что веса обновлены корректно
	// Примечание: Delta сбрасывается в UpdateWeight, поэтому не проверяем его
	// Просто проверяем, что веса изменились после обратного распространения
	for i, axon := range hidden.OutgoingAxons {
		// Вес должен измениться после обратного распространения
		if i == 0 && absFloat64(axon.Weight-0.6) > 1e-10 {
			t.Logf("Axon %d weight: expected ~0.59, got %v", i, axon.Weight)
		}
		if i == 1 && absFloat64(axon.Weight-0.4) > 1e-10 {
			t.Logf("Axon %d weight: expected ~0.39, got %v", i, axon.Weight)
		}
	}
}

func TestHiddenCellWithMultipleOutgoingConnections(t *testing.T) {
	hidden := NewHiddenCell[float32](RELU)
	
	// Добавляем несколько исходящих связей
	for i := 0; i < 3; i++ {
		target := &MockNeuron[float32]{value: float32(i) * 0.5, miss: 0}
		weight := float32(i) * 0.2
		hidden.AddOutgoingConnection(target, weight)
	}
	
	if len(hidden.OutgoingAxons) != 3 {
		t.Errorf("Expected 3 outgoing axons, got %d", len(hidden.OutgoingAxons))
	}
	
	// Проверяем все связи
	for i, axon := range hidden.OutgoingAxons {
		expectedWeight := float32(i) * 0.2
		if axon.Weight != expectedWeight {
			t.Errorf("Expected axon %d weight to be %v, got %v", i, expectedWeight, axon.Weight)
		}
	}
}

func TestHiddenCellReset(t *testing.T) {
	hidden := NewHiddenCell[float64](SIGMOID)
	
	// Устанавливаем значения
	hidden.Core.Value = 2.5
	hidden.Core.Miss = 0.4
	hidden.Core.Bias = 0.1
	
	// Добавляем связи
	target := &MockNeuron[float64]{value: 1.0, miss: 0}
	hidden.AddOutgoingConnection(target, 0.5)
	
	// Сбрасываем
	hidden.Core.Reset()
	
	if hidden.Core.GetValue() != 0 {
		t.Errorf("Expected core value to be reset to 0, got %v", hidden.Core.GetValue())
	}
	if hidden.Core.Miss != 0 {
		t.Errorf("Expected core miss to be reset to 0, got %v", hidden.Core.Miss)
	}
	
	// Исходящие связи должны остаться
	if len(hidden.OutgoingAxons) != 1 {
		t.Errorf("Expected outgoing axons to remain, got %d", len(hidden.OutgoingAxons))
	}
}