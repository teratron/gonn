package nn

import (
	"testing"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	round "github.com/teratron/gonn/pkg/utils"
)

func init() {
	GetMaxIteration = func() int { return 1 }
}

func Test_getMaxIteration(t *testing.T) {
	t.Run("MaxIteration", func(t *testing.T) {
		if got := getMaxIteration(); got != MaxIteration {
			t.Errorf("getMaxIteration() = %v, want %v", got, MaxIteration)
		}
	})
}

func TestNN_Train(t *testing.T) {
	type args struct {
		input  []float64
		target []float64
	}
	tests := []struct {
		name string
		args
		gave      *NN
		wantLoss  float64
		wantCount int
	}{
		{
			name: "#1",
			args: args{[]float64{.2, .3}, []float64{.2}},
			gave: &NN{
				ActivationMode: activation.SIGMOID,
				LossMode:       loss.MSE,
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
				input:          make(pkg.Float1Type, 2),
				target:         make(pkg.Float1Type, 1),
				isInit:         true,
			},
			wantLoss:  .123683,
			wantCount: 1,
		},
		{
			name:      "#2_no_input",
			args:      args{input: []float64{}},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name:      "#3_no_target",
			args:      args{[]float64{.2}, []float64{}},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#4_error_len_input",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &NN{
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#5_error_len_target",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &NN{
				lenInput:  1,
				lenOutput: 2,
				isInit:    true,
			},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#6_not_init",
			args: args{[]float64{.2, .3}, []float64{.3}},
			gave: &NN{
				Bias:         true,
				HiddenLayers: []uint{2},
				LossLimit:    .95,
			},
			wantLoss:  .0025,
			wantCount: 1,
		},
		{
			name: "#7_NaN",
			args: args{[]float64{2358925515.52, .66, .81}, []float64{-.13, .2}},
			gave: &NN{ActivationMode: activation.TANH},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() { _ = recover() }()
			gotCount, gotLoss := tt.gave.Train(tt.input, tt.target)
			if round.Round(gotLoss, round.ROUND, 6) != tt.wantLoss {
				t.Errorf("Train() gotLoss = %f, wantLoss %f", gotLoss, tt.wantLoss)
			}
			if gotCount != tt.wantCount {
				t.Errorf("Train() gotCount = %d, wantCount %d", gotCount, tt.wantCount)
			}
		})
	}
}
