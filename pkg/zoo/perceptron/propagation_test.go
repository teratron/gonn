package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/internal/pkg/math"
	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
)

func TestNN_calcNeuron(t *testing.T) {
	tests := []struct {
		name  string
		input *[]float64
		got   *NN
		want  [][]*neuron
	}{
		{
			name:  "#1",
			input: &[]float64{.2},
			got: &NN{
				Activation: params.LEAKYRELU,
				Weights: pkg.Float3Type{
					{
						{.1},
					},
				},
				neuron: [][]*neuron{
					{
						&neuron{},
					},
				},
				lenInput: 1,
			},
			want: [][]*neuron{
				{
					{.02, 0},
				},
			},
		},
		{
			name:  "#2",
			input: &[]float64{.2, .3},
			got: &NN{
				Activation: params.TANH,
				Weights: pkg.Float3Type{
					{
						{.1, .1},
						{.1, .1},
					},
				},
				neuron: [][]*neuron{
					{
						&neuron{},
						&neuron{},
					},
				},
				lenInput: 2,
			},
			want: [][]*neuron{
				{
					{.049958, 0},
					{.049958, 0},
				},
			},
		},
		{
			name:  "#3",
			input: &[]float64{.2, .3},
			got: &NN{
				Activation: params.SIGMOID,
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
				lenInput: 2,
			},
			want: [][]*neuron{
				{
					{.53743, 0},
					{.53743, 0},
				},
				{
					{.551686, 0},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.got.calcNeuron(tt.input)
			for i, v := range tt.got.neuron {
				for j, n := range v {
					tt.got.neuron[i][j].value = pkg.FloatType(math.Round(float64(n.value), math.ModeRound, 6))
					tt.got.neuron[i][j].miss = pkg.FloatType(math.Round(float64(n.miss), math.ModeRound, 6))
				}
			}
			if !reflect.DeepEqual(tt.got.neuron, tt.want) {
				t.Errorf("calcNeuron()\ngot:\t%v\nwant:\t%v", tt.got.neuron, tt.want)
			}
		})
	}
}

func TestNN_calcLoss(t *testing.T) {
	tests := []struct {
		name   string
		target *[]float64
		gave   *NN
		want   float64
	}{
		{
			name:   "#1_RMSE",
			target: &[]float64{.2},
			gave: &NN{
				Activation: params.LEAKYRELU,
				Loss:       params.RMSE,
				neuron: [][]*neuron{
					{
						{.5516861990955205, 0},
					},
				},
				lenOutput:      1,
				lastLayerIndex: 0,
			},
			want: .351686,
		},
		{
			name:   "#2_ARCTAN",
			target: &[]float64{.2, .3},
			gave: &NN{
				Activation: params.TANH,
				Loss:       params.ARCTAN,
				neuron: [][]*neuron{
					{
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
				},
				lenOutput:      2,
				lastLayerIndex: 0,
			},
			want: .080124,
		},
		{
			name:   "#3_MSE",
			target: &[]float64{.2},
			gave: &NN{
				Activation: params.SIGMOID,
				Loss:       params.MSE,
				neuron: [][]*neuron{
					{
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
					{
						{.5516861990955205, 0},
					},
				},
				lenOutput:      1,
				lastLayerIndex: 1,
			},
			want: .123683,
		},
		{
			name:   "#4_AVG",
			target: &[]float64{.2},
			gave: &NN{
				Activation: params.LINEAR,
				Loss:       params.AVG,
				neuron: [][]*neuron{
					{
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
					{
						{.5516861990955205, 0},
					},
				},
				lenOutput:      1,
				lastLayerIndex: 1,
			},
			want: .351686,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.calcLoss(tt.target); math.Round(got, math.ModeRound, 6) != tt.want {
				t.Errorf("calcLoss() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestNN_calcMiss(t *testing.T) {
	tests := []struct {
		name string
		got  *NN
		want [][]*neuron
	}{
		{
			name: "#1",
			got: &NN{
				Activation: params.SIGMOID,
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
						{.53743, 0},
						{.53743, 0},
					},
					{
						{.551686, .167181},
					},
				},
				lastLayerIndex: 1,
			},
			want: [][]*neuron{
				{
					{.53743, .016718},
					{.53743, .016718},
				},
				{
					{.551686, .167181},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.got.calcMiss()
			for i, v := range tt.got.neuron {
				for j, n := range v {
					tt.got.neuron[i][j].value = pkg.FloatType(math.Round(float64(n.value), math.ModeRound, 6))
					tt.got.neuron[i][j].miss = pkg.FloatType(math.Round(float64(n.miss), math.ModeRound, 6))
				}
			}
			if !reflect.DeepEqual(tt.got.neuron, tt.want) {
				t.Errorf("calcNeuron()\ngot:\t%v\nwant:\t%v", tt.got.neuron, tt.want)
			}
		})
	}
}

func TestNN_updWeight(t *testing.T) {
	tests := []struct {
		name  string
		input *[]float64
		got   *NN
		want  pkg.Float3Type
	}{
		{
			name:  "#1",
			input: &[]float64{.2},
			got: &NN{
				Rate: .3,
				Weights: pkg.Float3Type{
					{
						{.1},
					},
				},
				neuron: [][]*neuron{
					{
						{.5516861990955205, -.003516861990955205},
					},
				},
				lenInput: 1,
			},
			want: pkg.Float3Type{
				{
					{.094725},
				},
			},
		},
		{
			name:  "#2",
			input: &[]float64{.2, .3},
			got: &NN{
				Activation: params.TANH,
				Rate:       .3,
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
						{.5374298453437496, .004156099350080159},
						{.5374298453437496, .004156099350080159},
					},
					{
						{.5516861990955205, .167180851026932},
					},
				},
				lenInput: 2,
			},
			want: pkg.Float3Type{
				{
					{.100177, .100266, .100887},
					{.100177, .100266, .100887},
				},
				{
					{.118751, .118751, .134889},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.got.updWeight(tt.input)
			for i, v := range tt.got.Weights {
				for j, w := range v {
					for k, g := range w {
						tt.got.Weights[i][j][k] = pkg.FloatType(math.Round(float64(g), math.ModeRound, 6))
					}
				}
			}
			if !reflect.DeepEqual(tt.got.Weights, tt.want) {
				t.Errorf("updWeight()\ngot:\t%v\nwant:\t%v", tt.got.Weights, tt.want)
			}
		})
	}
}
