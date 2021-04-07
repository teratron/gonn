package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/params"
)

func TestNN_calcNeuron(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		got   *NN
		want  [][]*neuron
	}{
		{
			name:  "#1",
			input: []float64{.2},
			got: &NN{
				Activation: params.ModeLEAKYRELU,
				Weights: gonn.Float3Type{
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
					{.020000000000000004, 0},
				},
			},
		},
		{
			name:  "#2",
			input: []float64{.2, .3},
			got: &NN{
				Activation: params.ModeTANH,
				Weights: gonn.Float3Type{
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
					{.04995837495788001, 0},
					{.04995837495788001, 0},
				},
			},
		},
		{
			name:  "#3",
			input: []float64{.2, .3},
			got: &NN{
				Activation: params.ModeSIGMOID,
				Weights: gonn.Float3Type{
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
					{.5374298453437496, 0},
					{.5374298453437496, 0},
				},
				{
					{.5516861990955205, 0},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.calcNeuron(tt.input); !reflect.DeepEqual(tt.got.neuron, tt.want) {
				t.Errorf("calcNeuron()\ngot:\t%v\nwant:\t%v", tt.got.neuron, tt.want)
			}
		})
	}
}

func TestNN_calcLoss(t *testing.T) {
	tests := []struct {
		name   string
		target []float64
		gave   *NN
		want   float64
	}{
		{
			name:   "#1",
			target: []float64{.2},
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
			},
			want: .3516861990955205,
		},
		{
			name:   "#2",
			target: []float64{.2, .3},
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
			},
			want: .08012420394945846,
		},
		{
			name:   "#3",
			target: []float64{.2},
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
			},
			want: .1236831826342541,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.calcLoss(tt.target); got != tt.want {
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
			name: "1",
			got: &NN{
				Activation: params.ModeSIGMOID,
				Weights: gonn.Float3Type{
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
					{.5374298453437496, .004156099350080159},
					{.5374298453437496, .004156099350080159},
				},
				{
					{.5516861990955205, .167180851026932},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.calcMiss(); !reflect.DeepEqual(tt.got.neuron, tt.want) {
				t.Errorf("calcNeuron()\ngot:\t%v\nwant:\t%v", tt.got.neuron, tt.want)
			}
		})
	}
}

func TestNN_updWeight(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		got   *NN
		want  gonn.Float3Type
	}{
		{
			name:  "#1",
			input: []float64{.2},
			got: &NN{
				Rate: params.DefaultRate,
				Weights: gonn.Float3Type{
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
			want: gonn.Float3Type{
				{
					{.0997889882805427},
				},
			},
		},
		{
			name:  "#2",
			input: []float64{.2, .3},
			got: &NN{
				Rate: params.DefaultRate,
				Weights: gonn.Float3Type{
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
			want: gonn.Float3Type{
				{
					{.10024936596100481, .10037404894150723, .10124682980502406},
					{.10024936596100481, .10037404894150723, .10124682980502406},
				},
				{
					{.12695439367355216, .12695439367355216, .1501542553080796},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.updWeight(tt.input); !reflect.DeepEqual(tt.got.Weights, tt.want) {
				t.Errorf("updWeight()\ngot:\t%v\nwant:\t%v", tt.got.Weights, tt.want)
			}
		})
	}
}
