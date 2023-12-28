package utils

import (
	"math"
	"testing"
)

func TestRound2(t *testing.T) {
	pi := 3.14159265358979323846264
	//p := math.Pi
	type args struct {
		f    float64
		mode uint8
		prec uint
	}
	tests := []struct {
		name string
		args
		want float64
	}{
		{
			name: "#1_ROUND",
			args: args{pi, ROUND, 6},
			want: 3.141593,
		},
		{
			name: "#2_FLOOR",
			args: args{pi, FLOOR, 6},
			want: 3.141592,
		},
		{
			name: "#3_CEIL",
			args: args{pi, CEIL, 6},
			want: 3.141593,
		},
		{
			name: "#4_NaN",
			args: args{pi, 255, 6},
			want: math.NaN(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Round(tt.f, tt.mode, tt.prec)
			if (!math.IsNaN(got) && got != tt.want) || (math.IsNaN(got) && !math.IsNaN(tt.want)) {
				t.Errorf("Round() = %v, want %v", got, tt.want)
			}
		})
	}
}
