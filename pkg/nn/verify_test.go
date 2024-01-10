package nn

import (
	"testing"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

func TestNN_Verify(t *testing.T) {
	type args struct {
		input  []float64
		target []float64
	}
	tests := []struct {
		name string
		args
		gave *NN
		want float64
	}{
		{
			name: "#1",
			args: args{[]float64{.2, .3}, []float64{.2}},
			gave: &NN{
				ActivationMode: 255, // default params.ModeSIGMOID
				LossMode:       255, // default params.ModeMSE
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
			want: .123683,
		},
		{
			name: "#2_no_input",
			args: args{input: []float64{}},
			want: -1,
		},
		{
			name: "#3_no_target",
			args: args{[]float64{.2}, []float64{}},
			want: -1,
		},
		{
			name: "#4_error_len_input",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &NN{
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
			want: -1,
		},
		{
			name: "#5_error_len_target",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &NN{
				lenInput:  1,
				lenOutput: 2,
				isInit:    true,
			},
			want: -1,
		},
		{
			name: "#6_not_init",
			args: args{[]float64{.2, .3}, []float64{.3}},
			gave: &NN{
				Bias:         true,
				HiddenLayers: []uint{2},
			},
			want: .0025,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.gave.Verify(tt.input, tt.target)
			if utils.Round(got, utils.ROUND, 6) != tt.want {
				t.Errorf("Verify() = %f, want %f", got, tt.want)
			}
		})
	}
}
