package params

import (
	"github.com/teratron/gonn/pkg/nn"
	"strconv"
	"testing"
)

func Test_getRandFloat(t *testing.T) {
	want := [3]nn.FloatType{-.5, 0, .5}

	for i := range want {
		t.Run("#"+strconv.Itoa(i+1), func(t *testing.T) {
			if got := getRandFloat(); got < want[0] || got == want[1] || got > want[2] {
				t.Errorf("getRandFloat() = %.3f", got)
			}
		})
	}
}
