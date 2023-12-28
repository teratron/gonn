package perceptron

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"

	"github.com/teratron/gonn/pkg/utils"
)

var testJSON = filepath.Join("..", "..", "testdata", "perceptron.json")

func init() {
	utils.GetRandFloat = func() pkg.FloatType { return .5 }
}

func TestNN_Init(t *testing.T) {
	testFile := &utils.FileJSON{Name: testJSON}
	testGot := &NN{}
	_ = testFile.Decode(testGot)
	testNN := &NN{
		Name:           Name,
		Bias:           true,
		HiddenLayer:    []uint{2},
		ActivationMode: activation.SIGMOID,
		LossMode:       loss.MSE,
		LossLimit:      .1,
		Rate:           .3,
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
		isInit:         true,
		config:         testFile,
		weights: pkg.Float3Type{
			{
				{0, 0, 0},
				{0, 0, 0},
			},
			{
				{0, 0, 0},
			},
		},
		input:  make(pkg.Float1Type, 2),
		target: make(pkg.Float1Type, 1),
		output: make([]float64, 1),
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
			got:  testGot,
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
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.Init(tt.gave...); !reflect.DeepEqual(tt.got, tt.want) {
				t.Errorf("Init()\ngot:\t%v\nwant:\t%v", tt.got, tt.want)
			}
		})
	}
}

func TestNN_initFromNew(t *testing.T) {
	r := utils.GetRandFloat()
	tests := []struct {
		name string
		got  *NN
		want *NN
	}{
		{
			name: "#1",
			got: &NN{
				HiddenLayer:    []uint{0},
				ActivationMode: activation.SIGMOID,
			},
			want: &NN{
				Bias:           false,
				HiddenLayer:    []uint{0},
				ActivationMode: activation.SIGMOID,
				Weights: pkg.Float3Type{
					{
						{r, r},
						{r, r},
					},
				},
				neurons: [][]*neuron{
					{
						&neuron{},
						&neuron{},
					},
				},
				lenInput:       2,
				lenOutput:      2,
				lastLayerIndex: 0,
				isInit:         true,
				weights: pkg.Float3Type{
					{
						{0, 0},
						{0, 0},
					},
				},
				input:  make(pkg.Float1Type, 2),
				target: make(pkg.Float1Type, 2),
				output: make([]float64, 2),
			},
		},
		{
			name: "#2",
			got: &NN{
				HiddenLayer: []uint{2},
			},
			want: &NN{
				Bias:        true,
				HiddenLayer: []uint{2},
				Weights: pkg.Float3Type{
					{
						{r, r, r},
						{r, r, r},
					},
					{
						{r, r, r},
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
				isInit:         true,
				weights: pkg.Float3Type{
					{
						{0, 0, 0},
						{0, 0, 0},
					},
					{
						{0, 0, 0},
					},
				},
				input:  make(pkg.Float1Type, 2),
				target: make(pkg.Float1Type, 1),
				output: make([]float64, 1),
			},
		},
	}

	for _, tt := range tests {
		tt.got.Bias = tt.want.Bias
		_ = copy(tt.got.HiddenLayer, tt.want.HiddenLayer)
		t.Run(tt.name, func(t *testing.T) {
			tt.got.initFromNew(tt.want.lenInput, tt.want.lenOutput)
			tt.got.initCompletion()
			if !reflect.DeepEqual(tt.got, tt.want) {
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
				Weights: pkg.Float3Type{
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
				lenInput:       2,
				lenOutput:      2,
				lastLayerIndex: 0,
				isInit:         true,
				weights: pkg.Float3Type{
					{
						{0, 0},
						{0, 0},
					},
				},
				input:  make(pkg.Float1Type, 2),
				target: make(pkg.Float1Type, 2),
				output: make([]float64, 2),
			},
		},
		{
			name: "#2",
			got:  &NN{},
			want: &NN{
				Bias:        true,
				HiddenLayer: []uint{2},
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
				isInit:         true,
				weights: pkg.Float3Type{
					{
						{0, 0, 0},
						{0, 0, 0},
					},
					{
						{0, 0, 0},
					},
				},
				input:  make(pkg.Float1Type, 2),
				target: make(pkg.Float1Type, 1),
				output: make([]float64, 1),
			},
		},
	}

	for _, tt := range tests {
		tt.got.Weights = pkg.DeepCopy(tt.want.Weights)
		t.Run(tt.name, func(t *testing.T) {
			tt.got.initFromWeight()
			tt.got.initCompletion()
			if !reflect.DeepEqual(tt.got, tt.want) {
				t.Errorf("initFromWeight()\ngot:\t%v\nwant:\t%v", tt.got, tt.want)
			}
		})
	}
}
