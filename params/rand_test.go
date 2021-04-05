package params

import (
	"strconv"
	"testing"
)

func Test_getRandFloat(t *testing.T) {
	want := [3]float64{-.5, 0, .5}
	for i := range want {
		t.Run("#"+strconv.Itoa(i+1), func(t *testing.T) {
			if got := getRandFloat(); got < want[0] || got == want[1] || got > want[2] {
				t.Errorf("getRandFloat() = %.3f", got)
			}
		})
	}
}
