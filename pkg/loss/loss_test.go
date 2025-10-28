package loss

import (
	"math"
	"testing"
)

func TestLossFunctions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		mode      LossType
		predicted float64
		target    float64
		expected  float64
		delta     float64 // tolerance for floating point comparison
	}{
		// Test MSE
		{"MSE Zero", MSE, 1.0, 1.0, 0.0, 1e-9},
		{"MSE Positive", MSE, 2.0, 1.0, 1.0, 1e-9},
		{"MSE Negative", MSE, 0.0, 1.0, 1.0, 1e-9},

		// Test MAE
		{"MAE Zero", MAE, 1.0, 1.0, 0.0, 1e-9},
		{"MAE Positive", MAE, 2.0, 1.0, 1.0, 1e-9},
		{"MAE Negative", MAE, 0.0, 1.0, 1.0, 1e-9},

		// Test AVG (same as MAE)
		{"AVG Zero", AVG, 1.0, 1.0, 0.0, 1e-9},
		{"AVG Positive", AVG, 2.0, 1.0, 1.0, 1e-9},
		{"AVG Negative", AVG, 0.0, 1.0, 1.0, 1e-9},

		// Test RMSE
		{"RMSE Zero", RMSE, 1.0, 1.0, 0.0, 1e-9},
		{"RMSE Positive", RMSE, 2.0, 1.0, 1.0, 1e-9},
		{"RMSE Negative", RMSE, 0.0, 1.0, 1.0, 1e-9},

		// Test ARCTAN
		{"ARCTAN Zero", ARCTAN, 0.0, 0.0, 0.0, 1e-9},
		{"ARCTAN Positive", ARCTAN, 1.0, 0.0, math.Atan(1.0), 1e-9},
		{"ARCTAN Negative", ARCTAN, 0.0, 1.0, math.Atan(-1.0), 1e-9},

		// Test BCE (with values that won't cause log(0))
		{"BCE Different", BCE, 0.8, 0.9, 0.361773, 1e-4}, // -(0.9*log(0.8) + 0.1*log(0.2))
		{"BCE Small", BCE, 0.2, 0.1, 0.361773, 1e-4},     // -(0.1*log(0.2) + 0.9*log(0.8))

		// Test MAPE
		{"MAPE Zero", MAPE, 1.0, 1.0, 0.0, 1e-6},
		{"MAPE Positive", MAPE, 2.0, 1.0, 100.0, 1e-6}, // (1-2)/1 = -1, abs = 1, *100 = 100
		{"MAPE Negative", MAPE, 0.0, 1.0, 100.0, 1e-6}, // (1-0)/1 = 1, *100 = 100

		// Test MSLE
		{"MSLE Zero", MSLE, 0.0, 0.0, 0.0, 1e-6}, // log(1) - log(1) = 0, 0^2 = 0
		{"MSLE Positive", MSLE, 1.0, 0.0, math.Pow(math.Log(2)-math.Log(1), 2), 1e-6},

		// Test HUBER with delta=1.0
		{"HUBER Small Diff", HUBER, 1.0, 1.1, 0.5 * 0.1 * 0.1, 1e-6}, // Within delta=1
		{"HUBER Large Diff", HUBER, 0.0, 2.0, 1.5, 1e-6},             // Outside delta=1: 1*2 - 0.5*1*1 = 1.5
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Loss(tt.predicted, tt.target, tt.mode)
			if math.Abs(result-tt.expected) > tt.delta {
				t.Errorf("Loss(%f, %f, %v) = %f, want %f", tt.predicted, tt.target, tt.mode, result, tt.expected)
			}
		})
	}
}

func TestLossVector(t *testing.T) {
	t.Parallel()

	// Test MSE vector
	predicted := []float64{1.0, 2.0, 3.0}
	target := []float64{1.0, 2.0, 3.0}
	result := LossVector(predicted, target, MSE)
	if result > 1e-9 {
		t.Errorf("LossVector(MSE) with same vectors should be ~0, got %f", result)
	}

	// Test different vectors
	predicted = []float64{1.0, 2.0}
	target = []float64{2.0, 3.0}
	result = LossVector(predicted, target, MSE)
	expected := (1.0 + 1.0) / 2.0 // (1^2 + 1^2) / 2
	if math.Abs(result-expected) > 1e-9 {
		t.Errorf("LossVector(MSE) = %f, want %f", result, expected)
	}
}

func TestEdgeCases(t *testing.T) {
	t.Parallel()

	// Test with very small values
	result := Loss(1e-10, 0.0, MSE)
	if result > 1e-18 {
		t.Errorf("MSE of very small values should be tiny, got %f", result)
	}

	// Test with very large values
	result = Loss(1e10, 0.0, MSE)
	if result != 1e20 {
		t.Errorf("MSE of large values should be exact, got %f", result)
	}
}

func BenchmarkLoss(b *testing.B) {
	b.Run("MSE", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Loss(1.5, 1.0, MSE)
		}
	})

	b.Run("MAE", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Loss(1.5, 1.0, MAE)
		}
	})

	b.Run("BCE", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Loss(0.7, 0.8, BCE)
		}
	})
}
