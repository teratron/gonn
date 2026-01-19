package loss

import (
	"testing"
)

func TestLossTypeString(t *testing.T) {
	tests := []struct {
		mode Type
		want string
	}{
		{MSE, "MSE"},
		{MAE, "MAE"},
		{CCE, "CROSS_ENTROPY"},
		{CROSS_ENTROPY, "CROSS_ENTROPY"},
		{BCE, "BCE"},
		{MAPE, "MAPE"},
		{MSLE, "MSLE"},
		{KLD, "KLD"},
		{COSINE, "COSINE"},
		{POISSON, "POISSON"},
		{HINGE, "HINGE"},
		{SQ_HINGE, "SQ_HINGE"},
		{CAT_HINGE, "CAT_HINGE"},
		{LOG_COSH, "LOG_COSH"},
		{HUBER, "HUBER"},
		{AVG, "AVG"},
		{RMSE, "RMSE"},
		{ARCTAN, "ARCTAN"},
		{Type(99), "Unknown"}, // Test unknown value
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.mode.String(); got != tt.want {
				t.Errorf("Type.String() = %v, want %v", got, tt.want)
			}
		})
	}
}
