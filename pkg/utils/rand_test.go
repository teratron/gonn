package utils

import (
	"strconv"
	"testing"

	. "github.com/teratron/gonn/pkg"
)

func Test_getRandFloat(t *testing.T) {
	want := [3]pkg.FloatType{-.5, 0, .5}

	for i := range want {
		t.Run("#"+strconv.Itoa(i+1), func(t *testing.T) {
			if got := getRandFloat(); got < want[0] || got == want[1] || got > want[2] {
				t.Errorf("getRandFloat() = %.3f", got)
			}
		})
	}
}

func TestGetRandFloat(t *testing.T) {
	type testCase[T Floater] struct {
		name string
		want [3]T
	}
	tests := []testCase[float32]{
		// TODO: Add test cases.
		{
			name: "",
			want: [3]float32{-.5, 0, .5},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getRandFloat(); got != tt.want {
				t.Errorf("GetRandFloat() = %.3f, want %.3f", got, tt.want)
			}
		})
	}
}
