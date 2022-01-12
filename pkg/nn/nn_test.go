package nn

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg/arch"
)

func TestNew(t *testing.T) {
	testNN := arch.Get(arch.Perceptron)
	tests := []struct {
		name string
		gave []string
		want NeuralNetwork
	}{
		{
			name: "#1_warning_empty",
			gave: []string{},
			want: testNN,
		},
		{
			name: "#2_" + arch.Perceptron,
			gave: []string{arch.Perceptron},
			want: testNN,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := New(tt.gave...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}
