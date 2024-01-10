package nn

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
)

func TestNew(t *testing.T) {
	want := &NN{
		Name:           Name,
		HiddenLayer:    []uint{0},
		ActivationMode: activation.SIGMOID,
		LossMode:       loss.MSE,
		LossLimit:      .01,
		Rate:           .3,
	}

	t.Run(want.Name, func(t *testing.T) {
		if got := New(); !reflect.DeepEqual(got, want) {
			t.Errorf("perceptron()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}
