package perceptron

import (
	"reflect"
	"testing"

	"github.com/zigenzoog/gonn/pkg"
	"github.com/zigenzoog/gonn/pkg/params"
)

func init() {
	GetMaxIteration = func() int { return 1 }
	params.GetRandFloat = func() pkg.FloatType { return .5 }
}

func Test_getMaxIteration(t *testing.T) {
	t.Run("MaxIteration", func(t *testing.T) {
		if got := getMaxIteration(); got != MaxIteration {
			t.Errorf("getMaxIteration() = %v, want %v", got, MaxIteration)
		}
	})
}

func TestNew(t *testing.T) {
	want := &NN{
		Name:       Name,
		Activation: params.ModeSIGMOID,
		Loss:       params.ModeMSE,
		Limit:      .01,
		Rate:       pkg.FloatType(params.DefaultRate),
	}
	t.Run(want.Name, func(t *testing.T) {
		if got := New(); !reflect.DeepEqual(got, want) {
			t.Errorf("New()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}
