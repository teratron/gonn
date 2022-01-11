package hopfield

import (
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	want := &NN{
		Name: NAME,
	}
	t.Run(want.Name, func(t *testing.T) {
		if got := New(); !reflect.DeepEqual(got, want) {
			t.Errorf("New()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}
