package cell

import (
	"testing"
)

func TestBiasCellCreation(t *testing.T) {
	bias := NewBiasCell[float32]()
	
	if bias == nil {
		t.Errorf("BiasCell creation failed")
	}
	
	if bias.GetValue() != 1.0 {
		t.Errorf("Expected bias value to be 1.0, got %v", bias.GetValue())
	}
}

func TestBiasCellGetValue(t *testing.T) {
	bias := NewBiasCell[float64]()
	
	// Значение всегда должно быть 1.0
	if bias.GetValue() != 1.0 {
		t.Errorf("Expected bias value to be 1.0, got %v", bias.GetValue())
	}
}

func TestBiasCellWithDifferentTypes(t *testing.T) {
	// Тестируем с разными типами чисел
	t.Run("float32", func(t *testing.T) {
		bias := NewBiasCell[float32]()
		if bias.GetValue() != 1.0 {
			t.Errorf("Expected bias value to be 1.0 for float32, got %v", bias.GetValue())
		}
	})
	
	t.Run("float64", func(t *testing.T) {
		bias := NewBiasCell[float64]()
		if bias.GetValue() != 1.0 {
			t.Errorf("Expected bias value to be 1.0 for float64, got %v", bias.GetValue())
		}
	})
}