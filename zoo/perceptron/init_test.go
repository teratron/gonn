package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/params"
)

func TestNN_Init(t *testing.T) {
}

func TestNN_initFromNew(t *testing.T) {
	r := params.GetRandFloat()
	tests := []struct {
		name string
		got  *NN
		want *NN
	}{
		{
			name: "#1",
			got:  &NN{},
			want: &NN{
				Bias:   false,
				Hidden: []int{0},
				Weights: gonn.Float3Type{
					{
						{r, r},
						{r, r},
					},
				},
				neuron: [][]*neuron{
					{
						&neuron{},
						&neuron{},
					},
				},
				lenInput:       2,
				lenOutput:      2,
				lastLayerIndex: 0,
				isInit:         true,
			},
		},
		{
			name: "#2",
			got:  &NN{},
			want: &NN{
				Bias:   true,
				Hidden: []int{2},
				Weights: gonn.Float3Type{
					{
						{r, r, r},
						{r, r, r},
					},
					{
						{r, r, r},
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
				isInit:         true,
			},
		},
	}
	for _, tt := range tests {
		tt.got.Bias = tt.want.Bias
		tt.got.Hidden = tt.want.Hidden
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.initFromNew(tt.want.lenInput, tt.want.lenOutput); !reflect.DeepEqual(tt.got, tt.want) {
				t.Errorf("initFromNew()\ngot:\t%v\nwant:\t%v", tt.got, tt.want)
			}
		})
	}
}

func TestNN_initFromWeight(t *testing.T) {
	tests := []struct {
		name string
		got  *NN
		want *NN
	}{
		{
			name: "#1",
			got:  &NN{},
			want: &NN{
				Hidden: []int{0},
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
				lenInput:       2,
				lenOutput:      2,
				lastLayerIndex: 0,
				isInit:         true,
			},
		},
		{
			name: "#2",
			got:  &NN{},
			want: &NN{
				Bias:   true,
				Hidden: []int{2},
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
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
			},
		},
	}
	for _, tt := range tests {
		tt.got.Weights = tt.want.Weights
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.initFromWeight(); !reflect.DeepEqual(tt.got, tt.want) {
				t.Errorf("initFromWeight()\ngot:\t%v\nwant:\t%v", tt.got, tt.want)
			}
		})
	}
}
