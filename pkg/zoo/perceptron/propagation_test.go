package perceptron

import (
	"reflect"
	"testing"

	"github.com/zigenzoog/gonn/internal/pkg/math"
	"github.com/zigenzoog/gonn/pkg"
	"github.com/zigenzoog/gonn/pkg/params"
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
				Activation: params.ModeLEAKYRELU,
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
				input:    []float64{.2},
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
				Activation: params.ModeTANH,
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
				input:    []float64{.2, .3},
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
				Activation: params.ModeSIGMOID,
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
				input:    []float64{.2, .3},
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
		name string
		gave *NN
		want float64
	}{
		{
			name: "#1_RMSE",

			gave: &NN{
				Activation: params.ModeLEAKYRELU,
				Loss:       params.ModeRMSE,
				neuron: [][]*neuron{
					{
						{.5516861990955205, 0},
					},
				},
				lenOutput:      1,
				lastLayerIndex: 0,
				output:         []float64{.2},
			},
			want: .351686,
		},
		{
			name: "#2_ARCTAN",
			gave: &NN{
				Activation: params.ModeTANH,
				Loss:       params.ModeARCTAN,
				neuron: [][]*neuron{
					{
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
				},
				lenOutput:      2,
				lastLayerIndex: 0,
				output:         []float64{.2, .3},
			},
			want: .080124,
		},
		{
			name: "#3_MSE",
			gave: &NN{
				Activation: params.ModeSIGMOID,
				Loss:       params.ModeMSE,
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
				output:         []float64{.2},
			},
			want: .123683,
		},
		{
			name: "#4_AVG",
			gave: &NN{
				Activation: params.ModeLINEAR,
				Loss:       params.ModeAVG,
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
				output:         []float64{.2},
			},
			want: .351686,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.calcLoss(); math.Round(got, math.ModeRound, 6) != tt.want {
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
				Activation: params.ModeSIGMOID,
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
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
					{
						{.5516861990955205, .167180851026932},
					},
				},
				lastLayerIndex: 1,
			},
			want: [][]*neuron{
				{
					{.53743, .004156},
					{.53743, .004156},
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
		name string
		got  *NN
		want pkg.Float3Type
	}{
		{
			name: "#1",
			got: &NN{
				Rate: pkg.FloatType(params.DefaultRate),
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
				input:    []float64{.2},
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
				Activation: params.ModeTANH,
				Rate:       pkg.FloatType(params.DefaultRate),
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
				input:    []float64{.2, .3},
			},
			want: pkg.Float3Type{
				{
					{.100249, .100374, .101247},
					{.100249, .100374, .101247},
				},
				{
					{.126954, .126954, .150154},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.got.updWeight()
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
