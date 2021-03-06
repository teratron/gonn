package perceptron

import (
	"reflect"
	"testing"

	"github.com/zigenzoog/gonn/pkg"
	"github.com/zigenzoog/gonn/pkg/params"
)

func TestNN_NeuronBias(t *testing.T) {
	want := &NN{Bias: true}
	t.Run("true", func(t *testing.T) {
		if !want.NeuronBias() {
			t.Errorf("NeuronBias() = %t, want %t", false, true)
		}
	})
}

func TestNN_SetNeuronBias(t *testing.T) {
	want := &NN{}
	t.Run("true", func(t *testing.T) {
		if want.SetNeuronBias(true); !want.Bias {
			t.Errorf("SetNeuronBias() = %t, want %t", true, false)
		}
	})
}

func TestNN_HiddenLayer(t *testing.T) {
	tests := []struct {
		name string
		gave *NN
		want []int
	}{
		{
			name: "#1_nil",
			gave: &NN{Hidden: nil},
			want: []int{0},
		},
		{
			name: "#2_[]",
			gave: &NN{Hidden: []int{}},
			want: []int{0},
		},
		{
			name: "#3_[0]",
			gave: &NN{Hidden: []int{0}},
			want: []int{0},
		},
		{
			name: "#4_[3_2_1]",
			gave: &NN{Hidden: []int{3, 2, 1}},
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

func TestNN_SetHiddenLayer(t *testing.T) {
	got := &NN{}
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

func TestNN_ActivationMode(t *testing.T) {
	want := &NN{Activation: params.SIGMOID}
	t.Run("ModeSIGMOID", func(t *testing.T) {
		if got := want.ActivationMode(); got != want.Activation {
			t.Errorf("ActivationMode() = %d, want %d", got, want.Activation)
		}
	})
}

func TestNN_SetActivationMode(t *testing.T) {
	got := &NN{}
	want := params.LINEAR
	t.Run("ModeLINEAR", func(t *testing.T) {
		if got.SetActivationMode(want); got.Activation != want {
			t.Errorf("SetActivationMode() = %d, want %d", got.Activation, want)
		}
	})
}

func TestNN_LossMode(t *testing.T) {
	want := &NN{Loss: params.ARCTAN}
	t.Run("ModeARCTAN", func(t *testing.T) {
		if got := want.LossMode(); got != want.Loss {
			t.Errorf("LossMode() = %d, want %d", got, want.Loss)
		}
	})
}

func TestNN_SetLossMode(t *testing.T) {
	tests := []struct {
		name string
		got  *NN
		gave uint8
		want uint8
	}{
		{
			name: "#1_ModeARCTAN",
			got:  &NN{},
			gave: params.ARCTAN,
			want: params.ARCTAN,
		},
		{
			name: "#2_default",
			got:  &NN{},
			gave: 255,
			want: params.MSE,
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

func TestNN_LossLimit(t *testing.T) {
	want := &NN{Limit: .1}
	t.Run("0.1", func(t *testing.T) {
		if got := want.LossLimit(); got != want.Limit {
			t.Errorf("LossLimit() = %f, want %f", got, want.Limit)
		}
	})
}

func TestNN_SetLossLimit(t *testing.T) {
	got := &NN{}
	want := .01
	t.Run("0.01", func(t *testing.T) {
		if got.SetLossLimit(want); got.Limit != want {
			t.Errorf("SetLossLimit() = %f, want %f", got.Limit, want)
		}
	})
}

func TestNN_LearningRate(t *testing.T) {
	want := &NN{Rate: .3}
	t.Run("DefaultRate", func(t *testing.T) {
		if got := want.LearningRate(); got != float64(want.Rate) {
			t.Errorf("LearningRate() = %f, want %f", got, want.Rate)
		}
	})
}

func TestNN_SetLearningRate(t *testing.T) {
	tests := []struct {
		name string
		got  *NN
		gave float64
		want pkg.FloatType
	}{
		{
			name: "#1_rate",
			got:  &NN{},
			gave: .3,
			want: .3,
		},
		{
			name: "#2_overflow",
			got:  &NN{},
			gave: -.1,
			want: .3,
		},
		{
			name: "#3_overflow",
			got:  &NN{},
			gave: 1.1,
			want: .3,
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

func TestNN_Weight(t *testing.T) {
	tests := []struct {
		name string
		gave *NN
		want pkg.Float3Type
	}{
		{
			name: "#1_nil",
			gave: &NN{Weights: nil},
			want: nil,
		},
		{
			name: "#2_[]",
			gave: &NN{Weights: pkg.Float3Type{}},
			want: pkg.Float3Type{},
		},
		{
			name: "#3_[[[0.1_0.2_0.3]]]",
			gave: &NN{Weights: pkg.Float3Type{{{.1, .2, .3}}}},
			want: pkg.Float3Type{{{.1, .2, .3}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := *tt.gave.Weight().(*pkg.Float3Type); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Weight()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func TestNN_SetWeight(t *testing.T) {
	got := &NN{}
	tests := []struct {
		name string
		want pkg.Float3Type
	}{
		{
			name: "#1_nil",
			want: nil,
		},
		{
			name: "#2_[]",
			want: pkg.Float3Type{},
		},
		{
			name: "#3_[[[0.1_0.2_0.3]]]",
			want: pkg.Float3Type{{{.1, .2, .3}}},
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
