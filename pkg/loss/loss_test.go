package loss

import "testing"

func TestCheckLossMode(t *testing.T) {
	tests := []struct {
		name string
		gave Type
		want Type
	}{
		{
			name: "#1_MSE",
			gave: MSE,
			want: MSE,
		}, {
			name: "#2_RMSE",
			gave: RMSE,
			want: RMSE,
		}, {
			name: "#3_ARCTAN",
			gave: ARCTAN,
			want: ARCTAN,
		}, {
			name: "#4_AVG",
			gave: AVG,
			want: AVG,
		}, {
			name: "#5_overflow",
			gave: 255,
			want: MSE,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CheckLossMode(tt.gave); got != tt.want {
				t.Errorf("CheckLossMode() = %d, want %d", got, tt.want)
			}
		})
	}
}
