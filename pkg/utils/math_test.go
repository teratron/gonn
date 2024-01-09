package utils

import (
	"testing"

	"github.com/teratron/gonn/pkg"
)

func BenchmarkPow(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Pow(.5, 2)
	}
}

func TestPow(t *testing.T) {
	type args[T pkg.Floater] struct {
		x T
		y float64
	}
	type testCase[T pkg.Floater] struct {
		name string
		args args[T]
		want T
	}
	tests := []testCase[float32]{
		// TODO: Add test cases.
		{
			name: "#1",
			args: args[float32]{x: .5, y: 2},
			want: .1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Pow(tt.args.x, tt.args.y); got != tt.want {
				t.Errorf("Pow() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExp(t *testing.T) {
	type args[T pkg.Floater] struct {
		x T
	}
	type testCase[T pkg.Floater] struct {
		name string
		args args[T]
		want T
	}
	tests := []testCase[float32]{
		// TODO: Add test cases.
		{
			name: "#1",
			args: args[float32]{.5},
			want: .1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Exp(tt.args.x); got != tt.want {
				t.Errorf("Exp() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsNaN(t *testing.T) {
	type args[T pkg.Floater] struct {
		f T
	}
	type testCase[T pkg.Floater] struct {
		name string
		args args[T]
		want bool
	}
	tests := []testCase[float32]{
		// TODO: Add test cases.
		{
			name: "#1",
			args: args[float32]{.5},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsNaN(tt.args.f); got != tt.want {
				t.Errorf("IsNaN() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsInf(t *testing.T) {
	type args[T pkg.Floater] struct {
		f    T
		sign int
	}
	type testCase[T pkg.Floater] struct {
		name string
		args args[T]
		want bool
	}
	tests := []testCase[float32]{
		// TODO: Add test cases.
		{
			name: "#1",
			args: args[float32]{f: .5, sign: 1},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsInf(tt.args.f, tt.args.sign); got != tt.want {
				t.Errorf("IsInf() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRound(t *testing.T) {
	type args[T pkg.Floater] struct {
		x         T
		precision uint
	}
	type testCase[T pkg.Floater] struct {
		name string
		args args[T]
		want T
	}
	tests := []testCase[float64]{
		// TODO: Add test cases.
		{
			name: "#1",
			args: args[float64]{x: .5555555, precision: 3},
			want: .555,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Round(tt.args.x, tt.args.precision); got != tt.want {
				t.Errorf("Round() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFloor(t *testing.T) {
	type args[T pkg.Floater] struct {
		x         T
		precision uint
	}
	type testCase[T pkg.Floater] struct {
		name string
		args args[T]
		want T
	}
	tests := []testCase[float64]{
		// TODO: Add test cases.
		{
			name: "#1",
			args: args[float64]{x: .5555555, precision: 3},
			want: .555,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Floor(tt.args.x, tt.args.precision); got != tt.want {
				t.Errorf("Floor() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCeil(t *testing.T) {
	type args[T pkg.Floater] struct {
		x         T
		precision uint
	}
	type testCase[T pkg.Floater] struct {
		name string
		args args[T]
		want T
	}
	tests := []testCase[float64]{
		// TODO: Add test cases.
		{
			name: "#1",
			args: args[float64]{x: .5555555, precision: 3},
			want: .555,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Ceil(tt.args.x, tt.args.precision); got != tt.want {
				t.Errorf("Ceil() = %v, want %v", got, tt.want)
			}
		})
	}
}

/*func TestRound2(t *testing.T) {
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
}*/
