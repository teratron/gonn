package nn

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

func TestNN_Query(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		gave  *NN
		want  []float64
	}{
		{
			name:  "#1",
			input: []float64{.2, .3},
			gave: &NN{
				ActivationMode: activation.SIGMOID,
				Weights: pkg.Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neurons: [][]*neuron{
					{
						&neuron{},
						&neuron{},
					},
					{
						&neuron{},
					},
				},
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
				input:          make(pkg.Float1Type, 2),
				target:         make(pkg.Float1Type, 1),
				output:         make([]float64, 1),
			},
			want: []float64{.551686},
		},
		{
			name:  "#2_no_input",
			input: []float64{},
			want:  nil,
		},
		{
			name:  "#3_not_init",
			input: []float64{.1},
			gave:  &NN{isInit: false},
			want:  nil,
		},
		{
			name:  "#4_error_len_input",
			input: []float64{.1},
			gave: &NN{
				lenInput: 2,
				isInit:   true,
			},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.gave.Query(tt.input)
			for i, g := range got {
				got[i] = utils.Round(g, utils.ROUND, 6)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Query()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}
