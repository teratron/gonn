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
		name string
		got  *NN
		want [][]*neuron
	}{
		{
			name: "#1",
			got: &NN{
				ActivationMode: params.LEAKYRELU,
				Weight: pkg.Float3Type{
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
				input:    pkg.Float1Type{.2},
			},
			want: [][]*neuron{
				{
					{.02, 0},
				},
			},
		},
		{
			name: "#2",
			got: &NN{
				ActivationMode: params.TANH,
				Weight: pkg.Float3Type{
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
				input:    pkg.Float1Type{.2, .3},
			},
			want: [][]*neuron{
				{
					{.049958, 0},
					{.049958, 0},
				},
			},
		},
		{
			name: "#3",
			got: &NN{
				ActivationMode: params.SIGMOID,
				Weight: pkg.Float3Type{
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
				input:    pkg.Float1Type{.2, .3},
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
			tt.got.calcNeuron()
			for i, v := range tt.got.neuron {
				for j, n := range v {
					tt.got.neuron[i][j].value = pkg.FloatType(math.Round(float64(n.value), math.ROUND, 6))
					tt.got.neuron[i][j].miss = pkg.FloatType(math.Round(float64(n.miss), math.ROUND, 6))
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
		name string
		gave *NN
		want float64
	}{
		{
			name: "#1_RMSE",

			gave: &NN{
				ActivationMode: params.LEAKYRELU,
				LossMode:       params.RMSE,
				neuron: [][]*neuron{
					{
						{.5516861990955205, 0},
					},
				},
				lenOutput:      1,
				lastLayerIndex: 0,
				target:         pkg.Float1Type{.2},
			},
			want: .351686,
		},
		{
			name: "#2_ARCTAN",
			gave: &NN{
				ActivationMode: params.TANH,
				LossMode:       params.ARCTAN,
				neuron: [][]*neuron{
					{
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
				},
				lenOutput:      2,
				lastLayerIndex: 0,
				target:         pkg.Float1Type{.2, .3},
			},
			want: .080124,
		},
		{
			name: "#3_MSE",
			gave: &NN{
				ActivationMode: params.SIGMOID,
				LossMode:       params.MSE,
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
				target:         pkg.Float1Type{.2},
			},
			want: .123683,
		},
		{
			name: "#4_AVG",
			gave: &NN{
				ActivationMode: params.LINEAR,
				LossMode:       params.AVG,
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
				target:         pkg.Float1Type{.2},
			},
			want: .351686,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.calcLoss(); math.Round(got, math.ROUND, 6) != tt.want {
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
				ActivationMode: params.SIGMOID,
				Weight: pkg.Float3Type{
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
					tt.got.neuron[i][j].value = pkg.FloatType(math.Round(float64(n.value), math.ROUND, 6))
					tt.got.neuron[i][j].miss = pkg.FloatType(math.Round(float64(n.miss), math.ROUND, 6))
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
		name string
		got  *NN
		want pkg.Float3Type
	}{
		{
			name: "#1",
			got: &NN{
				Rate: .3,
				Weight: pkg.Float3Type{
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
				input:    pkg.Float1Type{.2},
			},
			want: pkg.Float3Type{
				{
					{.094725},
				},
			},
		},
		{
			name: "#2",
			got: &NN{
				ActivationMode: params.TANH,
				Rate:           .3,
				Weight: pkg.Float3Type{
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
				input:    pkg.Float1Type{.2, .3},
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
			tt.got.updateWeight()
			for i, v := range tt.got.Weight {
				for j, w := range v {
					for k, g := range w {
						tt.got.Weight[i][j][k] = pkg.FloatType(math.Round(float64(g), math.ROUND, 6))
					}
				}
			}
			if !reflect.DeepEqual(tt.got.Weight, tt.want) {
				t.Errorf("updateWeight()\ngot:\t%v\nwant:\t%v", tt.got.Weight, tt.want)
			}
		})
	}
}
