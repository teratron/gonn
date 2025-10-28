package loss

import "testing"

func TestCheckLossMode(t *testing.T) {
	type args struct {
		mode Type
	}
	tests := []struct {
		name string
		args
		want Type
	}{
		{
			name: "#1_MSE",
			args: args{MSE},
			want: MSE,
		}, {
			name: "#2_RMSE",
			args: args{RMSE},
			want: RMSE,
		}, {
			name: "#3_ARCTAN",
			args: args{ARCTAN},
			want: ARCTAN,
		}, {
			name: "#4_AVG",
			args: args{AVG},
			want: AVG,
		}, {
			name: "#5_overflow",
			args: args{255},
			want: MSE,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CheckLossMode(tt.args.mode); got != tt.want {
				t.Errorf("CheckLossMode() = %d, want %d", got, tt.want)
			}
		})
	}
}
