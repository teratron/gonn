package rate

import (
	"testing"

	"github.com/teratron/gonn/pkg"
)

func TestCheckLearningRate(t *testing.T) {
	type args[T pkg.Floater] struct {
		rate T
	}
	type testCase[T pkg.Floater] struct {
		name string
		args args[T]
		want T
	}
	tests := []testCase[float32]{
		{
			name: "#1_normal",
			args: args[float32]{.5},
			want: .5,
		}, {
			name: "#2_overflow",
			args: args[float32]{1.5},
			want: .3,
		}, {
			name: "#3_overflow",
			args: args[float32]{-.5},
			want: .3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CheckLearningRate(tt.args.rate); got != tt.want {
				t.Errorf("CheckLearningRate() = %v, want %v", got, tt.want)
			}
		})
	}
}
