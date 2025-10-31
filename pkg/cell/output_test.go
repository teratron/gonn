package cell

import (
	"testing"
)

// Вспомогательная функция для вычисления абсолютного значения
func absFloat64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestOutputCellCreation(t *testing.T) {
	output := NewOutputCell[float32](SIGMOID)
	
	if output == nil {
		t.Errorf("OutputCell creation failed")
	}
	
	if output.Core == nil {
		t.Errorf("Core should not be nil")
	}
	
	if output.Target != 0 {
		t.Errorf("Expected initial target to be 0, got %v", output.Target)
	}
	
	if output.HasTarget {
		t.Errorf("Expected HasTarget to be false initially")
	}
}

func TestOutputCellGetValue(t *testing.T) {
	output := NewOutputCell[float64](RELU)
	output.Core.Value = 1.5
	
	if output.GetValue() != 1.5 {
		t.Errorf("Expected value to be 1.5, got %v", output.GetValue())
	}
}

func TestOutputCellGetMiss(t *testing.T) {
	output := NewOutputCell[float32](SIGMOID)
	
	// Без целевого значения ошибка должна быть 0
	if output.GetMiss() != 0 {
		t.Errorf("Expected miss to be 0 without target, got %v", output.GetMiss())
	}
	
	// Устанавливаем целевое значение
	output.SetTarget(0.8)
	output.Core.Value = 0.5
	
	expectedMiss := float32(0.8 - 0.5) // 0.3
	if output.GetMiss() != expectedMiss {
		t.Errorf("Expected miss to be %v, got %v", expectedMiss, output.GetMiss())
	}
}

func TestOutputCellCalculateValue(t *testing.T) {
	output := NewOutputCell[float64](LINEAR)
	
	// Добавляем входящие связи
	neuron1 := &MockNeuron[float64]{value: 1.0, miss: 0}
	neuron2 := &MockNeuron[float64]{value: 2.0, miss: 0}
	
	output.Core.AddIncomingConnection(neuron1, 0.5)
	output.Core.AddIncomingConnection(neuron2, 0.3)
	output.Core.SetBias(0.1)
	
	result := output.CalculateValue()
	
	// Ожидаемое значение: 0.1 + 1.0*0.5 + 2.0*0.3 = 1.2
	expected := float64(1.2)
	if result != expected {
		t.Errorf("Expected calculated value to be %v, got %v", expected, result)
	}
}

func TestOutputCellCalculateWeight(t *testing.T) {
	output := NewOutputCell[float32](SIGMOID)
	output.Core.Value = 2.0
	
	result := output.CalculateWeight(0.5)
	
	// Ожидаемое значение: 0.5 * 2.0 = 1.0
	expected := float32(1.0)
	if result != expected {
		t.Errorf("Expected calculated weight to be %v, got %v", expected, result)
	}
}

func TestOutputCellForward(t *testing.T) {
	output := NewOutputCell[float64](RELU)
	
	// Добавляем входящие связи для корректной работы CalculateValue
	neuron := &MockNeuron[float64]{value: 1.0, miss: 0}
	output.Core.AddIncomingConnection(neuron, 0.5)
	output.Core.SetBias(1.0) // Устанавливаем bias для получения предсказуемого результата
	
	result := output.Forward()
	
	// Forward() вызывает CalculateValue(), который вычисляет значение на основе входящих связей
	expected := output.CalculateValue()
	if result != expected {
		t.Errorf("Expected forward result to be %v, got %v", expected, result)
	}
	
	if output.Core.Value != expected {
		t.Errorf("Expected core value to be updated to %v, got %v", expected, output.Core.Value)
	}
}

func TestOutputCellBackward(t *testing.T) {
	output := NewOutputCell[float32](SIGMOID)
	
	target := float32(0.8)
	result := output.Backward(target)
	
	// Проверяем, что целевое значение установлено
	if !output.HasTarget {
		t.Errorf("Expected HasTarget to be true")
	}
	if output.Target != target {
		t.Errorf("Expected target to be %v, got %v", target, output.Target)
	}
	
	// Проверяем, что ошибка вычислена
	expectedMiss := target - output.GetValue()
	if output.Core.Miss != expectedMiss {
		t.Errorf("Expected miss to be %v, got %v", expectedMiss, output.Core.Miss)
	}
	
	// Проверяем возвращаемое значение
	expected := output.CalculateWeight(expectedMiss)
	if result != expected {
		t.Errorf("Expected backward result to be %v, got %v", expected, result)
	}
}

func TestOutputCellSetTarget(t *testing.T) {
	output := NewOutputCell[float64](SIGMOID)
	
	target := float64(0.9)
	output.SetTarget(target)
	
	if output.Target != target {
		t.Errorf("Expected target to be %v, got %v", target, output.Target)
	}
	
	if !output.HasTarget {
		t.Errorf("Expected HasTarget to be true")
	}
}

func TestOutputCellGetTarget(t *testing.T) {
	output := NewOutputCell[float32](RELU)
	
	target := float32(0.7)
	output.SetTarget(target)
	
	if output.GetTarget() != target {
		t.Errorf("Expected target to be %v, got %v", target, output.GetTarget())
	}
}

func TestOutputCellClearTarget(t *testing.T) {
	output := NewOutputCell[float64](SIGMOID)
	
	// Устанавливаем целевое значение
	output.SetTarget(0.6)
	
	// Очищаем его
	output.ClearTarget()
	
	if output.HasTarget {
		t.Errorf("Expected HasTarget to be false after clear")
	}
	
	// Target может остаться прежним, но HasTarget должен быть false
	// Это нормально, так как мы только сбрасываем флаг
}

func TestOutputCellAddIncomingConnection(t *testing.T) {
	output := NewOutputCell[float32](SIGMOID)
	
	// Создаем исходный нейрон
	source := &MockNeuron[float32]{value: 1.0, miss: 0.1}
	
	output.AddIncomingConnection(source, 0.8)
	
	if len(output.Core.IncomingAxons) != 1 {
		t.Errorf("Expected 1 incoming axon, got %d", len(output.Core.IncomingAxons))
	}
	
	axon := output.Core.IncomingAxons[0]
	if axon.Target != source {
		t.Errorf("Expected axon target to be the source neuron")
	}
	if axon.Weight != 0.8 {
		t.Errorf("Expected axon weight to be 0.8, got %v", axon.Weight)
	}
}

func TestOutputCellGetError(t *testing.T) {
	output := NewOutputCell[float64](SIGMOID)
	
	// Без целевого значения ошибка должна быть 0
	if output.GetError() != 0 {
		t.Errorf("Expected error to be 0 without target, got %v", output.GetError())
	}
	
	// Устанавливаем целевое значение
	output.SetTarget(1.0)
	output.Core.Value = 0.3
	
	expectedError := float64(1.0 - 0.3) // 0.7
	if output.GetError() != expectedError {
		t.Errorf("Expected error to be %v, got %v", expectedError, output.GetError())
	}
}

func TestOutputCellIsCorrect(t *testing.T) {
	output := NewOutputCell[float32](SIGMOID)
	
	// Без целевого значения должно возвращать false
	if output.IsCorrect(0.1) {
		t.Errorf("Expected IsCorrect to return false without target")
	}
	
	// Устанавливаем целевое значение
	output.SetTarget(1.0)
	output.Core.Value = 0.95
	
	// В пределах допуска
	if !output.IsCorrect(0.1) {
		t.Errorf("Expected IsCorrect to return true within tolerance")
	}
	
	// Вне пределов допуска
	output.Core.Value = 0.8
	if output.IsCorrect(0.1) {
		t.Errorf("Expected IsCorrect to return false outside tolerance")
	}
}

func TestOutputCellGetSquaredError(t *testing.T) {
	output := NewOutputCell[float64](SIGMOID)
	
	// Без целевого значения квадрат ошибки должен быть 0
	if output.GetSquaredError() != 0 {
		t.Errorf("Expected squared error to be 0 without target, got %v", output.GetSquaredError())
	}
	
	// Устанавливаем целевое значение
	output.SetTarget(1.0)
	output.Core.Value = 0.7
	
	expectedSquaredError := float64((1.0 - 0.7) * (1.0 - 0.7)) // 0.09
	actualSquaredError := output.GetSquaredError()
	
	// Используем сравнение с допуском для чисел с плавающей точкой
	if absFloat64(actualSquaredError-expectedSquaredError) > 1e-10 {
		t.Errorf("Expected squared error to be %v, got %v", expectedSquaredError, actualSquaredError)
	}
}

func TestOutputCellReset(t *testing.T) {
	output := NewOutputCell[float32](RELU)
	
	// Устанавливаем значения
	output.Core.Value = 2.5
	output.Core.Miss = 0.4
	output.Core.Bias = 0.1
	output.SetTarget(1.0)
	
	// Сбрасываем
	output.Reset()
	
	if output.Core.GetValue() != 0 {
		t.Errorf("Expected core value to be reset to 0, got %v", output.Core.GetValue())
	}
	if output.Core.Miss != 0 {
		t.Errorf("Expected core miss to be reset to 0, got %v", output.Core.Miss)
	}
	if output.HasTarget {
		t.Errorf("Expected HasTarget to be false after reset")
	}
}

func TestOutputCellSetBias(t *testing.T) {
	output := NewOutputCell[float64](SIGMOID)
	
	bias := float64(0.5)
	output.SetBias(bias)
	
	if output.GetBias() != bias {
		t.Errorf("Expected bias to be %v, got %v", bias, output.GetBias())
	}
}

func TestOutputCellGetBias(t *testing.T) {
	output := NewOutputCell[float32](RELU)
	output.Core.Bias = 1.5
	
	if output.GetBias() != 1.5 {
		t.Errorf("Expected bias to be 1.5, got %v", output.GetBias())
	}
}