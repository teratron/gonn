package nn

import (
	"reflect"
	"testing"
)

func TestPerceptron(t *testing.T) {
	want := &perceptron{
		Name:       perceptronName,
		Activation: ModeSIGMOID,
		Loss:       ModeMSE,
		Limit:      .1,
		Rate:       floatType(DefaultRate),
	}
	t.Run("Default perceptron", func(t *testing.T) {
		if got := Perceptron(); !reflect.DeepEqual(got, want) {
			t.Errorf("Perceptron()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}

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
			t.Errorf("setName(%s), want %s", got.Name, want)
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
			t.Errorf("setStateInit(%t), want %t", true, false)
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
			t.Errorf("setNameJSON(%s), want %s", got.jsonName, want)
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
			t.Errorf("SetNeuronBias(%t), want %t", true, false)
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
	got := &perceptron{}
	want := ModeLINEAR
	t.Run("ModeLINEAR", func(t *testing.T) {
		if got.SetActivationMode(want); got.Activation != want {
			t.Errorf("SetActivationMode(%d), want %d", got.Activation, want)
		}
	})
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
			t.Errorf("SetLossMode(%d), want %d", got.Loss, want)
		}
	})
}

func Test_perceptron_LossLimit(t *testing.T) {
	want := &perceptron{Limit: .1}
	t.Run("0.1", func(t *testing.T) {
		if got := want.LossLimit(); got != want.Limit {
			t.Errorf("LossLimit() = %.3f, want %.3f", got, want.Limit)
		}
	})
}

func Test_perceptron_SetLossLimit(t *testing.T) {
	got := &perceptron{}
	want := .01
	t.Run("0.01", func(t *testing.T) {
		if got.SetLossLimit(want); got.Limit != want {
			t.Errorf("SetLossLimit(%.3f), want %.3f", got.Limit, want)
		}
	})
}

func Test_perceptron_LearningRate(t *testing.T) {
	want := &perceptron{Rate: floatType(DefaultRate)}
	t.Run("DefaultRate", func(t *testing.T) {
		if got := want.LearningRate(); got != float32(want.Rate) {
			t.Errorf("LearningRate() = %.3f, want %.3f", got, want.Rate)
		}
	})
}

func Test_perceptron_SetLearningRate(t *testing.T) {
	got := &perceptron{}
	want := DefaultRate
	t.Run("DefaultRate", func(t *testing.T) {
		if got.SetLearningRate(want); got.Rate != floatType(want) {
			t.Errorf("SetLearningRate(%.3f), want %.3f", got.Rate, want)
		}
	})
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
			got.SetWeight(tt.want)
			if !reflect.DeepEqual(got.Weights, tt.want) {
				t.Errorf("SetWeight()\ngot:\t%v\nwant:\t%v", got.Weights, tt.want)
			}
		})
	}
}

func Test_perceptron_initFromNew(t *testing.T) {
	r := floatType(.5)
	random := func() floatType {
		return r
	}
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
				lenInput:  2,
				lenOutput: 2,
				isInit:    true,
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
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
		},
	}
	for _, tt := range tests {
		tt.got.Bias = tt.want.Bias
		tt.got.Hidden = tt.want.Hidden
		tt.want.lastLayerIndex = len(tt.want.Hidden)
		if tt.want.lastLayerIndex > 0 && tt.want.Hidden[0] == 0 {
			tt.want.lastLayerIndex = 0
		}
		t.Run(tt.name, func(t *testing.T) {
			if tt.got.initFromNew(tt.want.lenInput, tt.want.lenOutput, random); !reflect.DeepEqual(tt.got, tt.want) {
				t.Errorf("initFromNew()\ngot:\t%v\nwant:\t%v", tt.got, tt.want)
			}
		})
	}
}

/*
func Test_perceptron_Query(t *testing.T) {
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
		input []float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantOutput []float64
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
			if gotOutput := p.Query(tt.args.input); !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("Query() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}

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

func Test_perceptron_Train(t *testing.T) {
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
		input  []float64
		target [][]float64
	}
	tests := []struct {
		name      string
		fields    fields
		args      args
		wantLoss  float64
		wantCount int
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
			gotLoss, gotCount := p.Train(tt.args.input, tt.args.target...)
			if gotLoss != tt.wantLoss {
				t.Errorf("Train() gotLoss = %v, want %v", gotLoss, tt.wantLoss)
			}
			if gotCount != tt.wantCount {
				t.Errorf("Train() gotCount = %v, want %v", gotCount, tt.wantCount)
			}
		})
	}
}

func Test_perceptron_Verify(t *testing.T) {
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
		input  []float64
		target [][]float64
	}
	tests := []struct {
		name     string
		fields   fields
		args     args
		wantLoss float64
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
			if gotLoss := p.Verify(tt.args.input, tt.args.target...); gotLoss != tt.wantLoss {
				t.Errorf("Verify() = %v, want %v", gotLoss, tt.wantLoss)
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

func Test_perceptron_calcLoss(t *testing.T) {
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
		target []float64
	}
	tests := []struct {
		name     string
		fields   fields
		args     args
		wantLoss float64
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
			if gotLoss := p.calcLoss(tt.args.target); gotLoss != tt.wantLoss {
				t.Errorf("calcLoss() = %v, want %v", gotLoss, tt.wantLoss)
			}
		})
	}
}

func Test_perceptron_calcMiss(t *testing.T) {
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
	tests := []struct {
		name   string
		fields fields
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

func Test_perceptron_calcNeuron(t *testing.T) {
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
		input []float64
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

func Test_perceptron_initFromWeight(t *testing.T) {
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
	tests := []struct {
		name   string
		fields fields
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

func Test_perceptron_updWeight(t *testing.T) {
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
		input []float64
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
}*/

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
