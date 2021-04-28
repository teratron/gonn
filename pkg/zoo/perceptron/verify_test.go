package perceptron

import (
	"testing"

	"github.com/zigenzoog/gonn/internal/pkg/math"
	"github.com/zigenzoog/gonn/pkg"
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
				Activation: 255, // default params.ModeSIGMOID
				Loss:       255, // default params.ModeMSE
				Weights: pkg.Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuron{
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
				input:          make([]float64, 2),
				output:         make([]float64, 1),
				isInit:         true,
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
				Bias:   true,
				Hidden: []int{2},
			},
			want: .0025,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.gave.Verify(tt.input, tt.target)
			if math.Round(got, math.ModeRound, 6) != tt.want {
				t.Errorf("Verify() = %f, want %f", got, tt.want)
			}
		})
	}
}
