package perceptron

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
	"reflect"
	"testing"

	"github.com/teratron/gonn/internal/pkg/math"
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
				ActivationMode: activation.LEAKYRELU,
				Weights: nn.Float3Type{
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
				input:    nn.Float1Type{.2},
				output:   make([]float64, 1),
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
				ActivationMode: activation.TANH,
				Weights: nn.Float3Type{
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
				input:    nn.Float1Type{.2, .3},
				output:   make([]float64, 2),
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
				ActivationMode: activation.SIGMOID,
				Weights: nn.Float3Type{
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
				input:    nn.Float1Type{.2, .3},
				output:   make([]float64, 2),
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
					tt.got.neurons[i][j].value = nn.FloatType(math.Round(float64(n.value), math.ROUND, 6))
					tt.got.neurons[i][j].miss = nn.FloatType(math.Round(float64(n.miss), math.ROUND, 6))
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
				ActivationMode: activation.LEAKYRELU,
				LossMode:       loss.RMSE,
				neurons: [][]*neuron{
					{
						{.5516861990955205, 0},
					},
				},
				lenOutput:      1,
				lastLayerIndex: 0,
				target:         nn.Float1Type{.2},
			},
			want: .351686,
		},
		{
			name: "#2_ARCTAN",
			gave: &NN{
				ActivationMode: activation.TANH,
				LossMode:       loss.ARCTAN,
				neurons: [][]*neuron{
					{
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
				},
				lenOutput:      2,
				lastLayerIndex: 0,
				target:         nn.Float1Type{.2, .3},
			},
			want: .080124,
		},
		{
			name: "#3_MSE",
			gave: &NN{
				ActivationMode: activation.SIGMOID,
				LossMode:       loss.MSE,
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
				target:         nn.Float1Type{.2},
			},
			want: .123683,
		},
		{
			name: "#4_AVG",
			gave: &NN{
				ActivationMode: activation.LINEAR,
				LossMode:       loss.AVG,
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
				target:         nn.Float1Type{.2},
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
				ActivationMode: activation.SIGMOID,
				Weights: nn.Float3Type{
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
					tt.got.neurons[i][j].value = nn.FloatType(math.Round(float64(n.value), math.ROUND, 6))
					tt.got.neurons[i][j].miss = nn.FloatType(math.Round(float64(n.miss), math.ROUND, 6))
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
		want nn.Float3Type
	}{
		{
			name: "#1",
			got: &NN{
				Rate: .3,
				Weights: nn.Float3Type{
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
				input:    nn.Float1Type{.2},
			},
			want: nn.Float3Type{
				{
					{.094725},
				},
			},
		},
		{
			name: "#2",
			got: &NN{
				ActivationMode: activation.TANH,
				Rate:           .3,
				Weights: nn.Float3Type{
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
				input:    nn.Float1Type{.2, .3},
			},
			want: nn.Float3Type{
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
			for i, v := range tt.got.Weights {
				for j, w := range v {
					for k, g := range w {
						tt.got.Weights[i][j][k] = nn.FloatType(math.Round(float64(g), math.ROUND, 6))
					}
				}
			}
			if !reflect.DeepEqual(tt.got.Weights, tt.want) {
				t.Errorf("updateWeights()\ngot:\t%v\nwant:\t%v", tt.got.Weights, tt.want)
			}
		})
	}
}
