package cell

import (
	"testing"
)

func TestInputCellCreation(t *testing.T) {
	data := float32(0.5)
	input := NewInputCell[float32](&data)
	
	if input == nil {
		t.Errorf("InputCell creation failed")
	}
	
	if input.GetValue() != 0 {
		t.Errorf("Expected initial value to be 0, got %v", input.GetValue())
	}
}

func TestInputCellGetValue(t *testing.T) {
	data := float64(1.5)
	input := NewInputCell[float64](&data)
	
	// Устанавливаем значение
	input.SetValue(2.0)
	
	if input.GetValue() != 2.0 {
		t.Errorf("Expected value to be 2.0, got %v", input.GetValue())
	}
}

func TestInputCellSetValue(t *testing.T) {
	input := NewInputCell[float32](nil)
	
	input.SetValue(3.14)
	
	if input.GetValue() != 3.14 {
		t.Errorf("Expected value to be 3.14, got %v", input.GetValue())
	}
	
	if !input.updated {
		t.Errorf("Expected updated flag to be true")
	}
}

func TestInputCellUpdateFromSource(t *testing.T) {
	data := float32(2.5)
	input := NewInputCell[float32](&data)
	
	// Изначально значение должно быть 0
	if input.GetValue() != 0 {
		t.Errorf("Expected initial value to be 0, got %v", input.GetValue())
	}
	
	// Обновляем из источника
	input.UpdateFromSource()
	
	if input.GetValue() != 2.5 {
		t.Errorf("Expected value to be 2.5 after update, got %v", input.GetValue())
	}
}

func TestInputCellGetMiss(t *testing.T) {
	input := NewInputCell[float64](nil)
	
	// Ошибка всегда должна быть 0, так как входные данные не обучаются
	if input.GetMiss() != 0 {
		t.Errorf("Expected miss to be 0, got %v", input.GetMiss())
	}
}

func TestInputCellCalculateValue(t *testing.T) {
	input := NewInputCell[float32](nil)
	input.SetValue(1.5)
	
	result := input.CalculateValue()
	if result != 1.5 {
		t.Errorf("Expected calculated value to be 1.5, got %v", result)
	}
}

func TestInputCellCalculateWeight(t *testing.T) {
	input := NewInputCell[float64](nil)
	
	// Вес всегда должен быть 0, так как входные данные не обучаются
	result := input.CalculateWeight(0.5)
	if result != 0 {
		t.Errorf("Expected calculated weight to be 0, got %v", result)
	}
}

func TestInputCellForward(t *testing.T) {
	data := float32(1.0)
	input := NewInputCell[float32](&data)
	
	// Первый вызов Forward должен обновить значение из источника
	result := input.Forward()
	if result != 1.0 {
		t.Errorf("Expected forward result to be 1.0, got %v", result)
	}
	
	// Изменяем исходные данные
	data = 2.0
	
	// Следующий вызов Forward НЕ должен обновить значение, так как updated = true
	result = input.Forward()
	if result != 1.0 {
		t.Errorf("Expected forward result to remain 1.0, got %v", result)
	}
}

func TestInputCellBackward(t *testing.T) {
	input := NewInputCell[float32](nil)
	
	// Обратное распространение должно возвращать 0
	result := input.Backward(0.8)
	if result != 0 {
		t.Errorf("Expected backward result to be 0, got %v", result)
	}
}

func TestInputCellIsInput(t *testing.T) {
	input := NewInputCell[float64](nil)
	
	if !input.IsInput() {
		t.Errorf("Expected IsInput to return true")
	}
}

func TestInputCellGetSource(t *testing.T) {
	data := float32(3.0)
	input := NewInputCell[float32](&data)
	
	source := input.GetSource()
	if source != &data {
		t.Errorf("Expected source to be the same pointer")
	}
}

func TestInputCellSetSource(t *testing.T) {
	input := NewInputCell[float32](nil)
	newData := float32(4.5)
	
	input.SetSource(&newData)
	
	source := input.GetSource()
	if source != &newData {
		t.Errorf("Expected source to be updated")
	}
	
	if input.updated {
		t.Errorf("Expected updated flag to be false after setting new source")
	}
}

func TestInputCellReset(t *testing.T) {
	input := NewInputCell[float64](nil)
	
	// Устанавливаем значение
	input.SetValue(5.0)
	
	// Сбрасываем
	input.Reset()
	
	if input.GetValue() != 0 {
		t.Errorf("Expected value to be reset to 0, got %v", input.GetValue())
	}
	
	if input.updated {
		t.Errorf("Expected updated flag to be false after reset")
	}
}

func TestInputCellSetData(t *testing.T) {
	data := float32(2.0)
	input := NewInputCell[float32](&data)
	
	// Устанавливаем данные напрямую
	input.SetData(6.0)
	
	// Проверяем значение в клетке
	if input.GetValue() != 6.0 {
		t.Errorf("Expected cell value to be 6.0, got %v", input.GetValue())
	}
	
	// Проверяем, что исходные данные также обновились
	if data != 6.0 {
		t.Errorf("Expected source data to be updated to 6.0, got %v", data)
	}
}

func TestInputCellForwardWithSource(t *testing.T) {
	data := float32(1.5)
	input := NewInputCell[float32](&data)
	
	// Первый вызов Forward должен обновить значение из источника
	result := input.Forward()
	if result != 1.5 {
		t.Errorf("Expected forward result to be 1.5, got %v", result)
	}
	
	// Изменяем исходные данные
	data = 2.5
	
	// Следующий вызов Forward НЕ должен обновить значение, так как updated = true
	result = input.Forward()
	if result != 1.5 {
		t.Errorf("Expected forward result to remain 1.5, got %v", result)
	}
}