package nn

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

const testStreamJSON = `
{
    "bias": true,
    "hiddenLayer": [
        2
    ],
    "activationMode": 3,
    "lossMode": 0,
    "lossLimit": 0.1,
    "rate": 0.3,
    "weights": [
        [
            [
                0.1,
                0.1,
                0.1
            ],
            [
                0.1,
                0.1,
                0.1
            ]
        ],
        [
            [
                0.1,
                0.1,
                0.1
            ]
        ]
    ]
}`

var testJSON = filepath.Join("..", "testdata", "perceptron.json")

func TestGet(t *testing.T) {
	testNN := &NN[float32]{
		Name:           Perceptron,
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
	}
	tests := []struct {
		name string
		gave string
		want pkg.NeuralNetwork
	}{
		{
			name: "#1_warning_empty",
			gave: "",
			want: nil,
		},
		{
			name: "#2_" + Perceptron,
			gave: Perceptron,
			want: New(),
		},
		{
			name: "#4_json_file",
			gave: testJSON,
			want: testNN,
		},
		{
			name: "#5_json_stream",
			gave: testStreamJSON,
			want: testNN,
		},
		{
			name: "#6_json_error_type",
			gave: ".json",
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.gave == testJSON {
				testNN.Init(&utils.FileJSON{Name: testJSON})
			}
			if tt.gave == testStreamJSON {
				testNN.Init(&utils.FileJSON{Name: ""})
			}
			if got := Get(tt.gave); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Get()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}
