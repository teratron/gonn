package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/params"
)

func init() {
	GetMaxIteration = func() int { return 1 }
	params.GetRandFloat = func() float64 { return .5 }
}

func TestPerceptron(t *testing.T) {
	want := &perceptron{
		Name:       Name,
		Activation: params.ModeSIGMOID,
		Loss:       params.ModeMSE,
		Limit:      .1,
		Rate:       params.DefaultRate,
	}
	t.Run(want.Name, func(t *testing.T) {
		if got := Perceptron(); !reflect.DeepEqual(got, want) {
			t.Errorf("Perceptron()\ngot:\t%v\nwant:\t%v", got, want)
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
			name: "#1_nil",
			gave: &perceptron{Hidden: nil},
			want: []int{0},
		},
		{
			name: "#2_[]",
			gave: &perceptron{Hidden: []int{}},
			want: []int{0},
		},
		{
			name: "#3_[0]",
			gave: &perceptron{Hidden: []int{0}},
			want: []int{0},
		},
		{
			name: "#4_[3_2_1]",
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
	got := &perceptron{}
	tests := []struct {
		name string
		gave []int
		want []int
	}{
		{
			name: "#1_nil",
			gave: nil,
			want: []int{0},
		},
		{
			name: "#2_[]",
			gave: []int{},
			want: []int{0},
		},
		{
			name: "#3_[0]",
			gave: []int{0},
			want: []int{0},
		},
		{
			name: "#4_[1_2_3]",
			gave: []int{1, 2, 3},
			want: []int{1, 2, 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got.SetHiddenLayer(tt.gave...); !reflect.DeepEqual(got.Hidden, tt.want) {
				t.Errorf("SetHiddenLayer()\ngot:\t%v\nwant:\t%v", got.Hidden, tt.want)
			}
		})
	}
}

func Test_perceptron_ActivationMode(t *testing.T) {
	want := &perceptron{Activation: params.ModeSIGMOID}
	t.Run("ModeSIGMOID", func(t *testing.T) {
		if got := want.ActivationMode(); got != want.Activation {
			t.Errorf("ActivationMode() = %d, want %d", got, want.Activation)
		}
	})
}

func Test_perceptron_SetActivationMode(t *testing.T) {
	got := &perceptron{}
	want := params.ModeLINEAR
	t.Run("ModeLINEAR", func(t *testing.T) {
		if got.SetActivationMode(want); got.Activation != want {
			t.Errorf("SetActivationMode() = %d, want %d", got.Activation, want)
		}
	})
}

func Test_perceptron_LossMode(t *testing.T) {
	want := &perceptron{Loss: params.ModeARCTAN}
	t.Run("ModeARCTAN", func(t *testing.T) {
		if got := want.LossMode(); got != want.Loss {
			t.Errorf("LossMode() = %d, want %d", got, want.Loss)
		}
	})
}

func Test_perceptron_SetLossMode(t *testing.T) {
	tests := []struct {
		name string
		got  *perceptron
		gave uint8
		want uint8
	}{
		{
			name: "#1_ModeARCTAN",
			got:  &perceptron{},
			gave: params.ModeARCTAN,
			want: params.ModeARCTAN,
		},
		{
			name: "#2_default",
			got:  &perceptron{},
			gave: 255,
			want: params.ModeMSE,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.SetLossMode(tt.gave); tt.got.Loss != tt.want {
				t.Errorf("SetLossMode() = %d, want %d", tt.got.Loss, tt.want)
			}
		})
	}
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
	want := &perceptron{Rate: params.DefaultRate}
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
			gave: params.DefaultRate,
			want: params.DefaultRate,
		},
		{
			name: "#2_default",
			got:  &perceptron{},
			gave: -.1,
			want: params.DefaultRate,
		},
		{
			name: "#3_default",
			got:  &perceptron{},
			gave: 1.1,
			want: params.DefaultRate,
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
		want gonn.Float3Type
	}{
		{
			name: "#1_nil",
			gave: &perceptron{Weights: nil},
			want: nil,
		},
		{
			name: "#2_[]",
			gave: &perceptron{Weights: gonn.Float3Type{}},
			want: gonn.Float3Type{},
		},
		{
			name: "#3_[[[0.1_0.2_0.3]]]",
			gave: &perceptron{Weights: gonn.Float3Type{{{.1, .2, .3}}}},
			want: gonn.Float3Type{{{.1, .2, .3}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := *tt.gave.Weight().(*gonn.Float3Type); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Weight()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_SetWeight(t *testing.T) {
	got := &perceptron{}
	tests := []struct {
		name string
		want gonn.Float3Type
	}{
		{
			name: "#1_nil",
			want: nil,
		},
		{
			name: "#2_[]",
			want: gonn.Float3Type{},
		},
		{
			name: "#3_[[[0.1_0.2_0.3]]]",
			want: gonn.Float3Type{{{.1, .2, .3}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got.SetWeight(tt.want); !reflect.DeepEqual(got.Weights, tt.want) {
				t.Errorf("SetWeight()\ngot:\t%v\nwant:\t%v", got.Weights, tt.want)
			}
		})
	}
}

/*func Test_perceptron_Read(t *testing.T) {
	gave := &perceptron{}
	tests := []struct {
		name    string
		reader  Reader
		wantErr error
	}{
		{
			name:    "#1_type_missing_read",
			reader:  Reader(nil),
			wantErr: fmt.Errorf("error"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErr := gave.Read(tt.reader)
			if (gotErr != nil && tt.wantErr == nil) || (gotErr == nil && tt.wantErr != nil) {
				t.Errorf("Read()\ngot error:\t%v\nwant error:\t%v", gotErr, tt.wantErr)
			}
		})
	}
}

func Test_perceptron_Write(t *testing.T) {
	existErr := fmt.Errorf("error")
	gave := &perceptron{}
	tests := []struct {
		name    string
		writer  []Writer
		wantErr error
	}{
		{
			name:    "#1",
			writer:  []Writer{JSON(defaultNameJSON)},
			wantErr: nil,
		},
		{
			name:    "#2_no_args",
			writer:  nil,
			wantErr: existErr,
		},
		{
			name:    "#3_type_missing_write",
			writer:  []Writer{nil},
			wantErr: existErr,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErr := gave.Write(tt.writer...)
			if len(tt.writer) > 0 {
				if w, ok := tt.writer[0].(jsonString); ok {
					defer func() {
						if err := os.Remove(string(w)); err != nil {
							t.Error(err)
						}
					}()
				}
			}
			if (gotErr != nil && tt.wantErr == nil) || (gotErr == nil && tt.wantErr != nil) {
				t.Errorf("Write()\ngot error:\t%v\nwant error:\t%v", gotErr, tt.wantErr)
			}
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
			name: "#1",
			args: args{[]float64{.2, .3}, []float64{.2}},
			gave: &perceptron{
				Activation: params.ModeSIGMOID,
				Loss:       params.ModeMSE,
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
			wantLoss:  .1236831826342541,
			wantCount: 1,
		},
		{
			name:      "#2_no_input",
			args:      args{input: []float64{}},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name:      "#3_no_target",
			args:      args{[]float64{.2}, []float64{}},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#4_error_len_input",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &perceptron{
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#5_error_len_target",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &perceptron{
				lenInput:  1,
				lenOutput: 2,
				isInit:    true,
			},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#6_not_init",
			args: args{[]float64{.2, .3}, []float64{.3}},
			gave: &perceptron{
				Bias:   true,
				Hidden: []int{2},
				Limit:  .95,
			},
			wantLoss:  .9025,
			wantCount: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotLoss, gotCount := tt.gave.Train(tt.input, tt.target)
			if gotLoss != tt.wantLoss {
				t.Errorf("Train() gotLoss = %f, wantLoss %f", gotLoss, tt.wantLoss)
			}
			if gotCount != tt.wantCount {
				t.Errorf("Train() gotCount = %d, wantCount %d", gotCount, tt.wantCount)
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
			args: args{[]float64{.2, .3}, []float64{.2}},
			gave: &perceptron{
				Activation: 255, // default ModeSIGMOID
				Loss:       255, // default ModeMSE
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
			want: .1236831826342541,
		},
		{
			name: "#2_no_input",
			args: args{input: []float64{}},
			want: -1,
		},
		{
			name: "#3_no_target",
			args: args{[]float64{.2}, []float64{}},
			want: -1,
		},
		{
			name: "#4_error_len_input",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &perceptron{
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
			want: -1,
		},
		{
			name: "#5_error_len_target",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &perceptron{
				lenInput:  1,
				lenOutput: 2,
				isInit:    true,
			},
			want: -1,
		},
		{
			name: "#6_not_init",
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
			input: []float64{.2, .3},
			gave: &perceptron{
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
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
			},
			want: []float64{.5516861990955205},
		},
		{
			name:  "#2_no_input",
			input: []float64{},
			want:  nil,
		},
		{
			name:  "#3_not_init",
			input: []float64{.1},
			gave:  &perceptron{isInit: false},
			want:  nil,
		},
		{
			name:  "#4_error_len_input",
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

func Test_perceptron_initFromNew(t *testing.T) {
	r := params.GetRandFloat()
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
			got:  &perceptron{},
			want: &perceptron{
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

func Test_perceptron_initFromWeight(t *testing.T) {
	tests := []struct {
		name string
		got  *perceptron
		want *perceptron
	}{
		{
			name: "#1",
			got:  &perceptron{},
			want: &perceptron{
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
			got:  &perceptron{},
			want: &perceptron{
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

func Test_perceptron_calcNeuron(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		got   *perceptron
		want  [][]*neuron
	}{
		{
			name:  "#1",
			input: []float64{.2},
			got: &perceptron{
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
			got: &perceptron{
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
			got: &perceptron{
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

func Test_perceptron_calcLoss(t *testing.T) {
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
			gave: &perceptron{
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
			gave: &perceptron{
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

func Test_perceptron_calcMiss(t *testing.T) {
	tests := []struct {
		name string
		got  *perceptron
		want [][]*neuron
	}{
		{
			name: "1",
			got: &perceptron{
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

func Test_perceptron_updWeight(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		got   *perceptron
		want  gonn.Float3Type
	}{
		{
			name:  "#1",
			input: []float64{.2},
			got: &perceptron{
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
			got: &perceptron{
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
