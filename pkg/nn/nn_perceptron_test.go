package nn

//go test -v ./...
//go test -v ./pkg/nn/nn_perceptron_test.go
import (
	"reflect"
	"testing"
)

/*func init() {
	randFloat = func() floatType {
		return .5
	}
}*/

func TestPerceptron(t *testing.T) {
	want := &perceptron{
		Name:       perceptronName,
		Activation: ModeSIGMOID,
		Loss:       ModeMSE,
		Limit:      .1,
		Rate:       floatType(DefaultRate),
		//random: getRandFloat,
	}
	//println(want)
	t.Run("Default perceptron", func(t *testing.T) {
		if got := Perceptron(); !reflect.DeepEqual(got, want) {
			t.Errorf("Perceptron()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}

func Test_perceptron_name(t *testing.T) {
	want := perceptronName
	p := &perceptron{Name: want}
	t.Run(want, func(t *testing.T) {
		if got := p.name(); got != want {
			t.Errorf("name() = %s, want %s", got, want)
		}
	})
}

func Test_perceptron_setName(t *testing.T) {
	got := perceptronName
	want := &perceptron{}
	want.setName(got)
	t.Run(got, func(t *testing.T) {
		if got != want.Name {
			t.Errorf("setName(%s), want %s", got, want.Name)
		}
	})
}

func Test_perceptron_stateInit(t *testing.T) {
	p := &perceptron{isInit: true}
	t.Run("true", func(t *testing.T) {
		if !p.stateInit() {
			t.Errorf("stateInit() = %t, want %t", false, true)
		}
	})
}

func Test_perceptron_setStateInit(t *testing.T) {
	want := &perceptron{}
	want.setStateInit(true)
	t.Run("true", func(t *testing.T) {
		if !want.isInit {
			t.Errorf("\nsetStateInit(%t), want %t", true, false)
		}
	})
}

func Test_perceptron_nameJSON(t *testing.T) {
	want := perceptronName + ".json"
	p := &perceptron{jsonName: want}
	t.Run(want, func(t *testing.T) {
		if got := p.nameJSON(); got != want {
			t.Errorf("nameJSON() = %s, want %s", got, want)
		}
	})
}

func Test_perceptron_setNameJSON(t *testing.T) {
	got := perceptronName + ".json"
	want := &perceptron{}
	want.setNameJSON(got)
	t.Run(got, func(t *testing.T) {
		if got != want.jsonName {
			t.Errorf("setNameJSON(%s), want %s", got, want.jsonName)
		}
	})
}

func Test_perceptron_NeuronBias(t *testing.T) {
	p := &perceptron{Bias: true}
	t.Run("true", func(t *testing.T) {
		if !p.NeuronBias() {
			t.Errorf("NeuronBias() = %t, want %t", false, true)
		}
	})
}

func Test_perceptron_SetNeuronBias(t *testing.T) {
	want := &perceptron{}
	want.SetNeuronBias(true)
	t.Run("true", func(t *testing.T) {
		if !want.Bias {
			t.Errorf("SetNeuronBias(%t), want %t", true, false)
		}
	})
}

func Test_perceptron_HiddenLayer(t *testing.T) {
	tests := []struct {
		name string
		got  *perceptron
		want []int
	}{
		{
			name: "nil",
			got:  &perceptron{Hidden: nil},
			want: []int{0},
		},
		{
			name: "[]",
			got:  &perceptron{Hidden: []int{}},
			want: []int{0},
		},
		{
			name: "[0]",
			got:  &perceptron{Hidden: []int{0}},
			want: []int{0},
		},
		{
			name: "[3,2,1]",
			got:  &perceptron{Hidden: []int{3, 2, 1}},
			want: []int{3, 2, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.got.HiddenLayer(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("HiddenLayer() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_SetHiddenLayer(t *testing.T) {
	tests := []struct {
		name string
		got  []int
		want []int
	}{
		{
			name: "nil",
			got:  nil,
			want: []int{0},
		},
		{
			name: "[]",
			got:  []int{},
			want: []int{0},
		},
		{
			name: "[0]",
			got:  []int{0},
			want: []int{0},
		},
		{
			name: "[1,2,3]",
			got:  []int{1, 2, 3},
			want: []int{1, 2, 3},
		},
	}
	p := &perceptron{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p.SetHiddenLayer(tt.got...)
			if !reflect.DeepEqual(p.Hidden, tt.want) {
				t.Errorf("SetHiddenLayer(%v), want %v", p.Hidden, tt.want)
			}
		})
	}
}

func Test_perceptron_ActivationMode(t *testing.T) {
	want := ModeSIGMOID
	p := &perceptron{Activation: want}
	t.Run("ModeSIGMOID", func(t *testing.T) {
		if got := p.ActivationMode(); got != want {
			t.Errorf("ActivationMode() = %d, want %d", got, want)
		}
	})
}

func Test_perceptron_SetActivationMode(t *testing.T) {
	got := ModeLINEAR
	want := &perceptron{}
	want.SetActivationMode(got)
	t.Run("ModeLINEAR", func(t *testing.T) {
		if got != want.Activation {
			t.Errorf("SetActivationMode(%d), want %d", got, want.Activation)
		}
	})
}

func Test_perceptron_LossMode(t *testing.T) {
	want := ModeARCTAN
	p := &perceptron{Loss: want}
	t.Run("ModeARCTAN", func(t *testing.T) {
		if got := p.LossMode(); got != want {
			t.Errorf("LossMode() = %d, want %d", got, want)
		}
	})
}

func Test_perceptron_SetLossMode(t *testing.T) {
	got := ModeMSE
	want := &perceptron{}
	want.SetLossMode(got)
	t.Run("ModeMSE", func(t *testing.T) {
		if got != want.Loss {
			t.Errorf("SetLossMode(%d), want %d", got, want.Loss)
		}
	})
}

func Test_perceptron_LossLimit(t *testing.T) {
	want := .1
	p := &perceptron{Limit: want}
	t.Run("0.1", func(t *testing.T) {
		if got := p.LossLimit(); got != want {
			t.Errorf("LossLimit() = %.3f, want %.3f", got, want)
		}
	})
}

func Test_perceptron_SetLossLimit(t *testing.T) {
	got := .01
	want := &perceptron{}
	want.SetLossLimit(got)
	t.Run("0.01", func(t *testing.T) {
		if got != want.Limit {
			t.Errorf("SetLossLimit(%.3f), want %.3f", got, want.Limit)
		}
	})
}

func Test_perceptron_LearningRate(t *testing.T) {
	want := DefaultRate
	p := &perceptron{Rate: floatType(want)}
	t.Run("DefaultRate", func(t *testing.T) {
		if got := p.LearningRate(); got != want {
			t.Errorf("LearningRate() = %.3f, want %.3f", got, want)
		}
	})
}

func Test_perceptron_SetLearningRate(t *testing.T) {
	got := DefaultRate
	want := &perceptron{}
	want.SetLearningRate(got)
	t.Run("DefaultRate", func(t *testing.T) {
		if got != float32(want.Rate) {
			t.Errorf("SetLearningRate(%.3f), want %.3f", got, want.Rate)
		}
	})
}

func Test_perceptron_Weight(t *testing.T) {
	tests := []struct {
		name string
		got  *perceptron
		want Float3Type
	}{
		{
			name: "nil",
			got:  &perceptron{Weights: nil},
			want: nil,
		},
		{
			name: "[]",
			got:  &perceptron{Weights: Float3Type{}},
			want: Float3Type{},
		},
		{
			name: "[[[0.1,0.2,0.3]]]",
			got:  &perceptron{Weights: Float3Type{{{.1, .2, .3}}}},
			want: Float3Type{{{.1, .2, .3}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := *tt.got.Weight().(*Float3Type); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Weight() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_SetWeight(t *testing.T) {
	tests := []struct {
		name string
		got  Float3Type
		want Float3Type
	}{
		{
			name: "nil",
			got:  nil,
			want: nil,
		},
		{
			name: "[]",
			got:  Float3Type{},
			want: Float3Type{},
		},
		{
			name: "[[[0.1,0.2,0.3]]]",
			got:  Float3Type{{{.1, .2, .3}}},
			want: Float3Type{{{.1, .2, .3}}},
		},
	}
	p := &perceptron{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p.SetWeight(tt.got)
			if !reflect.DeepEqual(p.Weights, tt.want) {
				t.Errorf("SetWeight(%v), want %v", p.Weights, tt.want)
			}
		})
	}
}

func Test_perceptron_initFromNew(t *testing.T) {
	random := func() floatType {
		return .5
	}
	tests := []struct {
		name string
		got  *perceptron
		want *perceptron
	}{
		{
			name: "",
			got:  &perceptron{},
			want: &perceptron{
				Bias:   true,
				Hidden: []int{2},
				Weights: Float3Type{
					{
						{.5, .5, .5},
						{.5, .5, .5},
					},
					{
						{.5, .5, .5},
					},
				},
				neuron: [][]*neuronPerceptron{
					{&neuronPerceptron{}, &neuronPerceptron{}},
					{&neuronPerceptron{}},
				},
				//random:    func() floatType { return .5 },
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
		},
	}
	for _, tt := range tests {
		tt.got.Bias = tt.want.Bias
		tt.got.Hidden = tt.want.Hidden
		//tt.got.random = tt.want.random
		tt.want.lastLayerIndex = len(tt.want.Hidden)
		t.Run(tt.name, func(t *testing.T) {
			tt.got.initFromNew(tt.want.lenInput, tt.want.lenOutput, random)

			//fmt.Println(len(tt.got.Weights), cap(tt.got.Weights), len(tt.want.Weights), cap(tt.want.Weights))
			/*println(tt.got.Weights, tt.want.Weights)
			for i := 0; i < len(tt.got.Weights); i++ {
				//fmt.Println(len(tt.got.Weights[i]), cap(tt.got.Weights[i]), len(tt.want.Weights[i]), cap(tt.want.Weights[i]))
				println(tt.got.Weights[i], tt.want.Weights[i])
				for j := 0; j < len(tt.got.Weights[i]); j++ {
					println(len(tt.got.Weights[i][j]), cap(tt.got.Weights[i][j]), len(tt.want.Weights[i][j]), cap(tt.want.Weights[i][j]))
				}
			}*/

			/*if lg, lw := len(tt.got.neuron), len(tt.want.neuron); lg == lw {
				//fmt.Println(lg, cap(tt.got.neuron), lw, cap(tt.want.neuron))
				for i := 0; i < lw; i++ {
					if llg, llw := len(tt.got.neuron[i]), len(tt.want.neuron[i]); llg == llw {
						//fmt.Println(i, llg, cap(tt.got.neuron[i]),llw, cap(tt.want.neuron[i]))
						tt.want.neuron[i] = tt.got.neuron[i]
					} else {
						t.Errorf("the number of elements in the neural layer does not match, got: %d, want: %d", llg, llw)
						break
					}
				}
			} else {
				t.Errorf("the number of neural layers does not match, got: %d, want: %d", lg, lw)
			}*/
			if !reflect.DeepEqual(tt.got, tt.want) {
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
//Parameter:      nil,
//Name:           perceptronName,
Bias:           true,
Hidden:         []int{1, 2, 3},
//Activation:     ModeSIGMOID,
//Loss:           ModeMSE,
//Limit:          .1,
//Rate:           floatType(DefaultRate),
Weights:        Float3Type{},
neuron:         [][]*neuronPerceptron{},
random:         func() floatType { return .5 },
lenInput:       2,
lenOutput:      2,
lastLayerIndex: 3,
isInit:         true,
//jsonName:       "perceptron.json",
}*/
