package perceptron

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
	"github.com/teratron/gonn/pkg/utils"
)

var testJSON = filepath.Join("..", "..", "testdata", "perceptron.json")

func init() {
	params.GetRandFloat = func() pkg.FloatType { return .5 }
}

func TestNN_Init(t *testing.T) {
	testFile := &utils.FileJSON{Name: testJSON}
	testNN := &NN{
		Name:           Name,
		Bias:           true,
		HiddenLayer:    []uint{2},
		ActivationMode: params.SIGMOID,
		LossMode:       params.MSE,
		LossLimit:      .1,
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
	}
	tests := []struct {
		name string
		gave []interface{}
		got  *NN
		want *NN
	}{
		{
			name: "#1_JSON",
			gave: []interface{}{testFile},
			got:  &NN{},
			want: testNN,
		},
		{
			name: "#2_error_type",
			gave: []interface{}{"test_error"},
			got:  &NN{},
			want: &NN{},
		},
		{
			name: "#3_empty_arguments",
			gave: []interface{}{},
			got:  &NN{},
			want: &NN{},
		},
	}

	for _, tt := range tests {
		tt.got.Weight = tt.want.Weight
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.Init(tt.gave...); !reflect.DeepEqual(tt.got, tt.want) {
				t.Errorf("Init()\ngot:\t%v\nwant:\t%v", tt.got, tt.want)
			}
		})
	}
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
			got: &NN{
				ActivationMode: params.SIGMOID,
			},
			want: &NN{
				Bias:           false,
				HiddenLayer:    []uint{0},
				ActivationMode: params.SIGMOID,
				Weight: pkg.Float3Type{
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
				input:          make(pkg.Float1Type, 2),
				target:         make(pkg.Float1Type, 2),
				isInit:         true,
			},
		},
		{
			name: "#2",
			got:  &NN{},
			want: &NN{
				Bias:        true,
				HiddenLayer: []uint{2},
				Weight: pkg.Float3Type{
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
				input:          make(pkg.Float1Type, 2),
				target:         make(pkg.Float1Type, 1),
				isInit:         true,
			},
		},
	}

	for _, tt := range tests {
		tt.got.Bias = tt.want.Bias
		tt.got.HiddenLayer = tt.want.HiddenLayer
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
				HiddenLayer: []uint{0},
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
				lenInput:       2,
				lenOutput:      2,
				lastLayerIndex: 0,
				input:          make(pkg.Float1Type, 2),
				target:         make(pkg.Float1Type, 2),
				isInit:         true,
			},
		},
		{
			name: "#2",
			got:  &NN{},
			want: &NN{
				Bias:        true,
				HiddenLayer: []uint{2},
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
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				input:          make(pkg.Float1Type, 2),
				target:         make(pkg.Float1Type, 1),
				isInit:         true,
			},
		},
	}

	for _, tt := range tests {
		tt.got.Weight = tt.want.Weight
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.initFromWeight(); !reflect.DeepEqual(tt.got, tt.want) {
				t.Errorf("initFromWeight()\ngot:\t%v\nwant:\t%v", tt.got, tt.want)
			}
		})
	}
}
