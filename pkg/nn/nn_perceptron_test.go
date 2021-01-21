package nn

import (
	"reflect"
	"testing"
)

func init() {
	maxIteration = func() int {
		return 1
	}
	randFloat = func() float64 {
		return .5
	}
}

/*func TestPerceptron(t *testing.T) {
	want := &perceptron{
		Name:       perceptronName,
		Activation: ModeSIGMOID,
		Loss:       ModeMSE,
		Limit:      .1,
		Rate:       DefaultRate,
	}
	t.Run("Default perceptron", func(t *testing.T) {
		if got := Perceptron(); !reflect.DeepEqual(got, want) {
			t.Errorf("Perceptron()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}*/

func Test_perceptron_name(t *testing.T) {
	want := &perceptron{Name: perceptronName}
	t.Run(want.Name, func(t *testing.T) {
		if got := want.name(); got != want.Name {
			t.Errorf("name() = %s, want %s", got, want.Name)
		}
	})
}

func Test_perceptron_setName(t *testing.T) {
	got := &perceptron{}
	want := perceptronName
	t.Run(want, func(t *testing.T) {
		if got.setName(want); got.Name != want {
			t.Errorf("setName() = %s, want %s", got.Name, want)
		}
	})
}

func Test_perceptron_stateInit(t *testing.T) {
	want := &perceptron{isInit: true}
	t.Run("true", func(t *testing.T) {
		if !want.stateInit() {
			t.Errorf("stateInit() = %t, want %t", false, true)
		}
	})
}

func Test_perceptron_setStateInit(t *testing.T) {
	want := &perceptron{}
	t.Run("true", func(t *testing.T) {
		if want.setStateInit(true); !want.isInit {
			t.Errorf("setStateInit() = %t, want %t", true, false)
		}
	})
}

func Test_perceptron_nameJSON(t *testing.T) {
	want := &perceptron{jsonName: perceptronName + ".json"}
	t.Run(want.jsonName, func(t *testing.T) {
		if got := want.nameJSON(); got != want.jsonName {
			t.Errorf("nameJSON() = %s, want %s", got, want.jsonName)
		}
	})
}

func Test_perceptron_setNameJSON(t *testing.T) {
	got := &perceptron{}
	want := perceptronName + ".json"
	t.Run(want, func(t *testing.T) {
		if got.setNameJSON(want); got.jsonName != want {
			t.Errorf("setNameJSON() = %s, want %s", got.jsonName, want)
		}
	})
}

func Test_perceptron_NeuronBias(t *testing.T) {
	want := &perceptron{Bias: true}
	t.Run("true", func(t *testing.T) {
		if !want.NeuronBias() {
			t.Errorf("NeuronBias() = %t, want %t", false, true)
		}
	})
}

func Test_perceptron_SetNeuronBias(t *testing.T) {
	want := &perceptron{}
	t.Run("true", func(t *testing.T) {
		if want.SetNeuronBias(true); !want.Bias {
			t.Errorf("SetNeuronBias() = %t, want %t", true, false)
		}
	})
}

func Test_perceptron_HiddenLayer(t *testing.T) {
	tests := []struct {
		name string
		gave *perceptron
		want []int
	}{
		{
			name: "nil",
			gave: &perceptron{Hidden: nil},
			want: []int{0},
		},
		{
			name: "[]",
			gave: &perceptron{Hidden: []int{}},
			want: []int{0},
		},
		{
			name: "[0]",
			gave: &perceptron{Hidden: []int{0}},
			want: []int{0},
		},
		{
			name: "[3_2_1]",
			gave: &perceptron{Hidden: []int{3, 2, 1}},
			want: []int{3, 2, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.HiddenLayer(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("HiddenLayer()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_SetHiddenLayer(t *testing.T) {
	tests := []struct {
		name string
		gave []int
		want []int
	}{
		{
			name: "nil",
			gave: nil,
			want: []int{0},
		},
		{
			name: "[]",
			gave: []int{},
			want: []int{0},
		},
		{
			name: "[0]",
			gave: []int{0},
			want: []int{0},
		},
		{
			name: "[1_2_3]",
			gave: []int{1, 2, 3},
			want: []int{1, 2, 3},
		},
	}
	got := &perceptron{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got.SetHiddenLayer(tt.gave...); !reflect.DeepEqual(got.Hidden, tt.want) {
				t.Errorf("SetHiddenLayer()\ngot:\t%v\nwant:\t%v", got.Hidden, tt.want)
			}
		})
	}
}

func Test_perceptron_ActivationMode(t *testing.T) {
	want := &perceptron{Activation: ModeSIGMOID}
	t.Run("ModeSIGMOID", func(t *testing.T) {
		if got := want.ActivationMode(); got != want.Activation {
			t.Errorf("ActivationMode() = %d, want %d", got, want.Activation)
		}
	})
}

func Test_perceptron_SetActivationMode(t *testing.T) {
	tests := []struct {
		name string
		got  *perceptron
		gave uint8
		want uint8
	}{
		{
			name: "#1_ModeLINEAR",
			got:  &perceptron{},
			gave: ModeLINEAR,
			want: ModeLINEAR,
		},
		{
			name: "#2_default",
			got:  &perceptron{},
			gave: 255,
			want: ModeSIGMOID,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.SetActivationMode(tt.gave); tt.got.Activation != tt.want {
				t.Errorf("SetActivationMode() = %d, want %d", tt.got.Activation, tt.want)
			}
		})
	}
}

func Test_perceptron_LossMode(t *testing.T) {
	want := &perceptron{Loss: ModeARCTAN}
	t.Run("ModeARCTAN", func(t *testing.T) {
		if got := want.LossMode(); got != want.Loss {
			t.Errorf("LossMode() = %d, want %d", got, want.Loss)
		}
	})
}

func Test_perceptron_SetLossMode(t *testing.T) {
	got := &perceptron{}
	want := ModeMSE
	t.Run("ModeMSE", func(t *testing.T) {
		if got.SetLossMode(want); got.Loss != want {
			t.Errorf("SetLossMode() = %d, want %d", got.Loss, want)
		}
	})
}

func Test_perceptron_LossLimit(t *testing.T) {
	want := &perceptron{Limit: .1}
	t.Run("0.1", func(t *testing.T) {
		if got := want.LossLimit(); got != want.Limit {
			t.Errorf("LossLimit() = %f, want %f", got, want.Limit)
		}
	})
}

func Test_perceptron_SetLossLimit(t *testing.T) {
	got := &perceptron{}
	want := .01
	t.Run("0.01", func(t *testing.T) {
		if got.SetLossLimit(want); got.Limit != want {
			t.Errorf("SetLossLimit() = %f, want %f", got.Limit, want)
		}
	})
}

func Test_perceptron_LearningRate(t *testing.T) {
	want := &perceptron{Rate: DefaultRate}
	t.Run("DefaultRate", func(t *testing.T) {
		if got := want.LearningRate(); got != want.Rate {
			t.Errorf("LearningRate() = %f, want %f", got, want.Rate)
		}
	})
}

func Test_perceptron_SetLearningRate(t *testing.T) {
	tests := []struct {
		name string
		got  *perceptron
		gave float64
		want float64
	}{
		{
			name: "#1_DefaultRate",
			got:  &perceptron{},
			gave: DefaultRate,
			want: DefaultRate,
		},
		{
			name: "#2_default",
			got:  &perceptron{},
			gave: -.1,
			want: DefaultRate,
		},
		{
			name: "#3_default",
			got:  &perceptron{},
			gave: 1.1,
			want: DefaultRate,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.SetLearningRate(tt.gave); tt.got.Rate != tt.want {
				t.Errorf("SetLearningRate() = %f, want %f", tt.got.Rate, tt.want)
			}
		})
	}
}

func Test_perceptron_Weight(t *testing.T) {
	tests := []struct {
		name string
		gave *perceptron
		want Float3Type
	}{
		{
			name: "nil",
			gave: &perceptron{Weights: nil},
			want: nil,
		},
		{
			name: "[]",
			gave: &perceptron{Weights: Float3Type{}},
			want: Float3Type{},
		},
		{
			name: "[[[0.1_0.2_0.3]]]",
			gave: &perceptron{Weights: Float3Type{{{.1, .2, .3}}}},
			want: Float3Type{{{.1, .2, .3}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := *tt.gave.Weight().(*Float3Type); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Weight()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_SetWeight(t *testing.T) {
	tests := []struct {
		name string
		want Float3Type
	}{
		{
			name: "nil",
			want: nil,
		},
		{
			name: "[]",
			want: Float3Type{},
		},
		{
			name: "[[[0.1_0.2_0.3]]]",
			want: Float3Type{{{.1, .2, .3}}},
		},
	}
	got := &perceptron{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got.SetWeight(tt.want); !reflect.DeepEqual(got.Weights, tt.want) {
				t.Errorf("SetWeight()\ngot:\t%v\nwant:\t%v", got.Weights, tt.want)
			}
		})
	}
}

func Test_perceptron_initFromNew(t *testing.T) {
	r := randFloat()
	tests := []struct {
		name string
		got  *perceptron
		want *perceptron
	}{
		{
			name: "#1",
			got:  &perceptron{},
			want: &perceptron{
				Bias:   false,
				Hidden: []int{0},
				Weights: Float3Type{
					{
						{r, r},
						{r, r},
					},
				},
				neuron: [][]*neuronPerceptron{
					{&neuronPerceptron{}, &neuronPerceptron{}},
				},
				lenInput:       2,
				lenOutput:      2,
				lastLayerIndex: 0,
				isInit:         true,
			},
		},
		{
			name: "#2",
			got:  &perceptron{},
			want: &perceptron{
				Bias:   true,
				Hidden: []int{2},
				Weights: Float3Type{
					{
						{r, r, r},
						{r, r, r},
					},
					{
						{r, r, r},
					},
				},
				neuron: [][]*neuronPerceptron{
					{&neuronPerceptron{}, &neuronPerceptron{}},
					{&neuronPerceptron{}},
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

/*func Test_perceptron_initFromWeight(t *testing.T) {
	tests := []struct {
		name   string
		fields *perceptron
	}{
		// TODO: Add test cases.
		{},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
		})
	}
}*/

func Test_perceptron_Train(t *testing.T) {
	type args struct {
		input  []float64
		target []float64
	}
	tests := []struct {
		name string
		args
		gave      *perceptron
		wantLoss  float64
		wantCount int
	}{
		{
			name: "#3",
			args: args{[]float64{.2, .3}, []float64{.2}},
			gave: &perceptron{
				Activation: ModeSIGMOID,
				Loss:       ModeMSE,
				Weights: Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
						&neuronPerceptron{},
					},
					{
						&neuronPerceptron{},
					},
				},
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
			},
			wantLoss:  .1236831826342541,
			wantCount: 1,
		},
	}
	for _, tt := range tests {
		tt.gave.lenInput = len(tt.input)
		tt.gave.lenOutput = len(tt.target)
		t.Run(tt.name, func(t *testing.T) {
			gotLoss, gotCount := tt.gave.Train(tt.input, tt.target)
			if gotLoss != tt.wantLoss {
				t.Errorf("Train() gotLoss = %f, want %f", gotLoss, tt.wantLoss)
			}
			if gotCount != tt.wantCount {
				t.Errorf("Train() gotCount = %d, want %d", gotCount, tt.wantCount)
			}
		})
	}
}

func Test_perceptron_Verify(t *testing.T) {
	type args struct {
		input  []float64
		target []float64
	}
	tests := []struct {
		name string
		args
		gave *perceptron
		want float64
	}{
		{
			name: "#1",
			args: args{[]float64{.3}, []float64{.2}},
			gave: &perceptron{
				Activation: ModeLEAKYRELU,
				Loss:       ModeRMSE,
				Weights: Float3Type{
					{
						{.1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
					},
				},
				lenInput:       1,
				lenOutput:      1,
				lastLayerIndex: 0,
				isInit:         true,
			},
			want: .17,
		},
		{
			name: "#2",
			args: args{[]float64{.3, .2}, []float64{.2, .3}},
			gave: &perceptron{
				Activation: ModeTANH,
				Loss:       ModeARCTAN,
				Weights: Float3Type{
					{
						{.1, .1},
						{.1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
						&neuronPerceptron{},
					},
				},
				lenInput:       2,
				lenOutput:      2,
				lastLayerIndex: 0,
				isInit:         true,
			},
			want: .041107040991293746,
		},
		{
			name: "#3",
			args: args{[]float64{.2, .3}, []float64{.2}},
			gave: &perceptron{
				Activation: ModeSIGMOID,
				Loss:       3, // error, need ModeMSE
				Weights: Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
						&neuronPerceptron{},
					},
					{
						&neuronPerceptron{},
					},
				},
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
			},
			want: .1236831826342541,
		},
		{
			name: "#4_no_input",
			args: args{input: []float64{}},
			want: -1,
		},
		{
			name: "#5_no_target",
			args: args{[]float64{.2}, []float64{}},
			want: -1,
		},
		{
			name: "#6_warning_len_input",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &perceptron{
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
			want: -1,
		},
		{
			name: "#7_warning_len_target",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &perceptron{
				lenInput:  1,
				lenOutput: 2,
				isInit:    true,
			},
			want: -1,
		},
		{
			name: "#8_not_init",
			args: args{[]float64{.2, .3}, []float64{.3}},
			gave: &perceptron{
				Bias:   true,
				Hidden: []int{2},
			},
			want: .9025,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.Verify(tt.input, tt.target); got != tt.want {
				t.Errorf("Verify() = %f, want %f", got, tt.want)
			}
		})
	}
}

func Test_perceptron_Query(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		gave  *perceptron
		want  []float64
	}{
		{
			name:  "#1",
			input: []float64{.2},
			gave: &perceptron{
				Activation: ModeLEAKYRELU,
				Weights: Float3Type{
					{
						{.1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
					},
				},
				lenInput:       1,
				lenOutput:      1,
				lastLayerIndex: 0,
				isInit:         true,
			},
			want: []float64{.020000000000000004},
		},
		{
			name:  "#2",
			input: []float64{.2, .3},
			gave: &perceptron{
				Activation: ModeTANH,
				Weights: Float3Type{
					{
						{.1, .1},
						{.1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
						&neuronPerceptron{},
					},
				},
				lenInput:       2,
				lenOutput:      2,
				lastLayerIndex: 0,
				isInit:         true,
			},
			want: []float64{.04995837495788001, .04995837495788001},
		},
		{
			name:  "#3",
			input: []float64{.2, .3},
			gave: &perceptron{
				Activation: ModeSIGMOID,
				Weights: Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
						&neuronPerceptron{},
					},
					{
						&neuronPerceptron{},
					},
				},
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
			},
			want: []float64{.5516861990955205},
		},
		{
			name:  "#4_no_input",
			input: []float64{},
			want:  nil,
		},
		{
			name:  "#5_not_init",
			input: []float64{.1},
			gave:  &perceptron{isInit: false},
			want:  nil,
		},
		{
			name:  "#6_warning_len_input",
			input: []float64{.1},
			gave: &perceptron{
				lenInput: 2,
				isInit:   true,
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.Query(tt.input); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Query()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

/*func Test_perceptron_calcNeuron(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		got   *perceptron
		want  [][]*neuronPerceptron
	}{
		{
			name:  "#1",
			input: []float64{.2},
			got: &perceptron{
				Activation: ModeLEAKYRELU,
				Weights: Float3Type{
					{
						{.1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
					},
				},
			},
			want: [][]*neuronPerceptron{
				{
					{.020000000000000004, 0},
				},
			},
		},
		{
			name:  "#2",
			input: []float64{.2, .3},
			got: &perceptron{
				Activation: ModeTANH,
				Weights: Float3Type{
					{
						{.1, .1},
						{.1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
						&neuronPerceptron{},
					},
				},
			},
			want: [][]*neuronPerceptron{
				{
					{.04995837495788001, 0},
					{.04995837495788001, 0},
				},
			},
		},
		{
			name:  "#3",
			input: []float64{.2, .3},
			got: &perceptron{
				Activation: ModeSIGMOID,
				Weights: Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						&neuronPerceptron{},
						&neuronPerceptron{},
					},
					{
						&neuronPerceptron{},
					},
				},
			},
			want: [][]*neuronPerceptron{
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
		tt.got.lenInput = len(tt.input)
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.calcNeuron(tt.input); !reflect.DeepEqual(tt.got.neuron, tt.want) {
				t.Errorf("calcNeuron()\ngot:\t%v\nwant:\t%v", tt.got.neuron, tt.want)
			}
		})
	}
}*/

/*func Test_perceptron_calcLoss(t *testing.T) {
	tests := []struct {
		name   string
		target []float64
		gave   *perceptron
		want   float64
	}{
		{
			name:   "#1",
			target: []float64{.2},
			gave: &perceptron{
				Activation: ModeLEAKYRELU,
				Loss:       ModeRMSE,
				neuron: [][]*neuronPerceptron{
					{
						{.5516861990955205, 0},
					},
				},
			},
			want: .3516861990955205,
		},
		{
			name:   "#2",
			target: []float64{.2, .3},
			gave: &perceptron{
				Activation: ModeTANH,
				Loss:       ModeARCTAN,
				neuron: [][]*neuronPerceptron{
					{
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
				},
			},
			want: .08012420394945846,
		},
		{
			name:   "#3",
			target: []float64{.2},
			gave: &perceptron{
				Activation: ModeSIGMOID,
				Loss:       ModeMSE,
				neuron: [][]*neuronPerceptron{
					{
						{.5374298453437496, 0},
						{.5374298453437496, 0},
					},
					{
						{.5516861990955205, 0},
					},
				},
				lastLayerIndex: 1,
			},
			want: .1236831826342541,
		},
	}
	for _, tt := range tests {
		tt.gave.lenOutput = len(tt.target)
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.calcLoss(tt.target); got != tt.want {
				t.Errorf("calcLoss() = %f, want %f", got, tt.want)
			}
		})
	}
}*/

func Test_perceptron_calcMiss(t *testing.T) {
	tests := []struct {
		name string
		got  *perceptron
		want [][]*neuronPerceptron
	}{
		{
			name: "3",
			got: &perceptron{
				Activation: ModeSIGMOID,
				Weights: Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
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
			want: [][]*neuronPerceptron{
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

func Test_perceptron_updWeight(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		got   *perceptron
		want  Float3Type
	}{
		{
			name:  "#1",
			input: []float64{.2},
			got: &perceptron{
				Rate: DefaultRate,
				Weights: Float3Type{
					{
						{.1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						{.5516861990955205, -.003516861990955205},
					},
				},
			},
			want: Float3Type{
				{
					{.0997889882805427},
				},
			},
		},
		{
			name:  "#2",
			input: []float64{.2, .3},
			got: &perceptron{
				Rate: DefaultRate,
				Weights: Float3Type{
					{
						{.1, .1},
						{.1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						{.5374298453437496, -.29901045414733873},
						{.5374298453437496, -.22404522894563828},
					},
				},
			},
			want: Float3Type{
				{
					{.08205937275115968, .07308905912673952},
					{.0865572862632617, .07983592939489256},
				},
			},
		},
		{
			name:  "#3",
			input: []float64{.2, .3},
			got: &perceptron{
				Rate: DefaultRate,
				Weights: Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuronPerceptron{
					{
						{.5374298453437496, .004156099350080159},
						{.5374298453437496, .004156099350080159},
					},
					{
						{.5516861990955205, .167180851026932},
					},
				},
			},
			want: Float3Type{
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
		tt.got.lenInput = len(tt.input)
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.updWeight(tt.input); !reflect.DeepEqual(tt.got.Weights, tt.want) {
				t.Errorf("updWeight()\ngot:\t%v\nwant:\t%v", tt.got.Weights, tt.want)
			}
		})
	}
}

/*
for _, n := range tt.got.neuron {
	for _, m := range n {
		fmt.Println(m)
	}
}
for _, n := range tt.got.Weights {
	for _, m := range n {
		fmt.Println(m)
	}
}
&perceptron{
Parameter:      nil,
Name:           perceptronName,
Bias:           true,
Hidden:         []int{1, 2, 3},
Activation:     ModeSIGMOID,
Loss:           ModeMSE,
Limit:          .1,
Rate:           floatType(DefaultRate),
Weights:        Float3Type{},
neuron:         [][]*neuronPerceptron{},
lenInput:       2,
lenOutput:      2,
lastLayerIndex: 3,
isInit:         true,
jsonName:       "perceptron.json",
}*/

/*
func Test_perceptron_Read(t *testing.T) {
	type fields struct {
		Parameter      Parameter
		Name           string
		Bias           bool
		Hidden         []int
		Activation     uint8
		Loss           uint8
		Limit          float64
		Rate           floatType
		Weights        Float3Type
		neuron         [][]*neuronPerceptron
		lenInput       int
		lenOutput      int
		lastLayerIndex int
		isInit         bool
		jsonName       string
	}
	type args struct {
		reader Reader
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Parameter:      tt.fields.Parameter,
				Name:           tt.fields.Name,
				Bias:           tt.fields.Bias,
				Hidden:         tt.fields.Hidden,
				Activation:     tt.fields.Activation,
				Loss:           tt.fields.Loss,
				Limit:          tt.fields.Limit,
				Rate:           tt.fields.Rate,
				Weights:        tt.fields.Weights,
				neuron:         tt.fields.neuron,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
				lastLayerIndex: tt.fields.lastLayerIndex,
				isInit:         tt.fields.isInit,
				jsonName:       tt.fields.jsonName,
			}
		})
	}
}

func Test_perceptron_Write(t *testing.T) {
	type fields struct {
		Parameter      Parameter
		Name           string
		Bias           bool
		Hidden         []int
		Activation     uint8
		Loss           uint8
		Limit          float64
		Rate           floatType
		Weights        Float3Type
		neuron         [][]*neuronPerceptron
		lenInput       int
		lenOutput      int
		lastLayerIndex int
		isInit         bool
		jsonName       string
	}
	type args struct {
		writer []Writer
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Parameter:      tt.fields.Parameter,
				Name:           tt.fields.Name,
				Bias:           tt.fields.Bias,
				Hidden:         tt.fields.Hidden,
				Activation:     tt.fields.Activation,
				Loss:           tt.fields.Loss,
				Limit:          tt.fields.Limit,
				Rate:           tt.fields.Rate,
				Weights:        tt.fields.Weights,
				neuron:         tt.fields.neuron,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
				lastLayerIndex: tt.fields.lastLayerIndex,
				isInit:         tt.fields.isInit,
				jsonName:       tt.fields.jsonName,
			}
		})
	}
}


*/

/*&perceptron{
Parameter:      nil,
Name:           perceptronName,
Bias:           true,
Hidden:         []int{1, 2, 3},
Activation:     ModeSIGMOID,
Loss:           ModeMSE,
Limit:          .1,
Rate:           floatType(DefaultRate),
Weights:        Float3Type{},
neuron:         [][]*neuronPerceptron{},
lenInput:       2,
lenOutput:      2,
lastLayerIndex: 3,
isInit:         true,
jsonName:       "perceptron.json",
}*/
