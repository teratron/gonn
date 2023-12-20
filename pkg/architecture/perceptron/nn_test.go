package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg/params"
)

func TestNew(t *testing.T) {
	want := &NN{
		Name:           Name,
		HiddenLayer:    []uint{0},
		ActivationMode: params.SIGMOID,
		LossMode:       params.MSE,
		LossLimit:      .01,
		Rate:           .3,
	}

	t.Run(want.Name, func(t *testing.T) {
		if got := New(); !reflect.DeepEqual(got, want) {
			t.Errorf("New()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}
