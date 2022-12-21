package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/internal/pkg/math"
	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
)

func TestNN_calcNeurons(t *testing.T) {
	tests := []struct {
		name string
		got  *NN
		want [][]*neuron
	}{
		{
			name: "#1",
			got: &NN{
				ActivationMode: params.LEAKYRELU,
				weights: pkg.Float3Type{
					{
						{.1},
					},
				},
				neurons: [][]*neuron{
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
				weights: pkg.Float3Type{
					{
						{.1, .1},
						{.1, .1},
					},
				},
				neurons: [][]*neuron{
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
				weights: pkg.Float3Type{
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
			tt.got.calcNeurons()
			for i, v := range tt.got.neurons {
				for j, n := range v {
					tt.got.neurons[i][j].value = pkg.FloatType(math.Round(float64(n.value), math.ROUND, 6))
					tt.got.neurons[i][j].miss = pkg.FloatType(math.Round(float64(n.miss), math.ROUND, 6))
				}
			}
			if !reflect.DeepEqual(tt.got.neurons, tt.want) {
				t.Errorf("calcNeurons()\ngot:\t%v\nwant:\t%v", tt.got.neurons, tt.want)
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
				neurons: [][]*neuron{
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
				neurons: [][]*neuron{
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
				neurons: [][]*neuron{
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
				neurons: [][]*neuron{
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
				weights: pkg.Float3Type{
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
			for i, v := range tt.got.neurons {
				for j, n := range v {
					tt.got.neurons[i][j].value = pkg.FloatType(math.Round(float64(n.value), math.ROUND, 6))
					tt.got.neurons[i][j].miss = pkg.FloatType(math.Round(float64(n.miss), math.ROUND, 6))
				}
			}
			if !reflect.DeepEqual(tt.got.neurons, tt.want) {
				t.Errorf("calcNeurons()\ngot:\t%v\nwant:\t%v", tt.got.neurons, tt.want)
			}
		})
	}
}

func TestNN_updateWeights(t *testing.T) {
	tests := []struct {
		name string
		got  *NN
		want pkg.Float3Type
	}{
		{
			name: "#1",
			got: &NN{
				Rate: .3,
				weights: pkg.Float3Type{
					{
						{.1},
					},
				},
				neurons: [][]*neuron{
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
				weights: pkg.Float3Type{
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
			tt.got.updateWeights()
			for i, v := range tt.got.weights {
				for j, w := range v {
					for k, g := range w {
						tt.got.weights[i][j][k] = pkg.FloatType(math.Round(float64(g), math.ROUND, 6))
					}
				}
			}
			if !reflect.DeepEqual(tt.got.weights, tt.want) {
				t.Errorf("updateWeights()\ngot:\t%v\nwant:\t%v", tt.got.weights, tt.want)
			}
		})
	}
}
