package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
)

func init() {
	GetMaxIteration = func() int { return 1 }
	params.GetRandFloat = func() pkg.FloatType { return .5 }
}

func Test_getMaxIteration(t *testing.T) {
	t.Run("MAXITERATION", func(t *testing.T) {
		if got := getMaxIteration(); got != MAXITERATION {
			t.Errorf("getMaxIteration() = %v, want %v", got, MAXITERATION)
		}
	})
}

func TestNew(t *testing.T) {
	want := &NN{
		Name:       NAME,
		Activation: params.SIGMOID,
		Loss:       params.MSE,
		Limit:      .01,
		Rate:       .3,
	}
	t.Run(want.Name, func(t *testing.T) {
		if got := New(); !reflect.DeepEqual(got, want) {
			t.Errorf("New()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}
