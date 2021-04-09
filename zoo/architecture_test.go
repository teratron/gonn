package zoo

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/params"
	"github.com/teratron/gonn/utils"
	"github.com/teratron/gonn/zoo/hopfield"
	"github.com/teratron/gonn/zoo/perceptron"
)

var (
	testJSON = filepath.Join("..", "testdata", "perceptron.json")
	testYAML = filepath.Join("..", "testdata", "perceptron.yml")
)

func TestGet(t *testing.T) {
	testNN := &perceptron.NN{
		Name:       Perceptron,
		Bias:       true,
		Hidden:     []int{2},
		Activation: params.ModeSIGMOID,
		Loss:       params.ModeMSE,
		Limit:      .1,
		Rate:       params.DefaultRate,
		Weights: gonn.Float3Type{
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
		want gonn.NeuralNetwork
	}{
		{
			name: "#1_warning_empty",
			gave: "",
			want: nil,
		},
		{
			name: "#2_" + Perceptron,
			gave: Perceptron,
			want: perceptron.New(),
		},
		{
			name: "#3_" + Hopfield,
			gave: Hopfield,
			want: hopfield.New(),
		},
		{
			name: "#4_json",
			gave: testJSON,
			want: testNN,
		},
		{
			name: "#5_yaml",
			gave: testYAML,
			want: testNN,
		},
		{
			name: "#6_error_type",
			gave: ".yaml",
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.want != nil {
				if nn, ok := tt.want.(*perceptron.NN); ok && len(nn.Weights) > 0 {
					nn.Init(utils.GetFileType(tt.gave))
				}
			}
			if got := Get(tt.gave); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Get()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}
