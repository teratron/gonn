package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
)

func TestNN_GetBias(t *testing.T) {
	want := &NN{Bias: true}
	t.Run("true", func(t *testing.T) {
		if !want.GetBias() {
			t.Errorf("GetBias() = %t, want %t", false, true)
		}
	})
}

func TestNN_SetBias(t *testing.T) {
	want := &NN{}
	t.Run("true", func(t *testing.T) {
		if want.SetBias(true); !want.Bias {
			t.Errorf("SetBias() = %t, want %t", true, false)
		}
	})
}

func TestNN_GetHiddenLayer(t *testing.T) {
	tests := []struct {
		name string
		gave *NN
		want []uint
	}{
		{
			name: "#1_nil",
			gave: &NN{HiddenLayer: nil},
			want: []uint{0},
		},
		{
			name: "#2_[]",
			gave: &NN{HiddenLayer: []uint{}},
			want: []uint{0},
		},
		{
			name: "#3_[0]",
			gave: &NN{HiddenLayer: []uint{0}},
			want: []uint{0},
		},
		{
			name: "#4_[3_2_1]",
			gave: &NN{HiddenLayer: []uint{3, 2, 1}},
			want: []uint{3, 2, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.GetHiddenLayer(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetHiddenLayer()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func TestNN_SetHiddenLayer(t *testing.T) {
	got := &NN{}
	tests := []struct {
		name string
		gave []uint
		want []uint
	}{
		{
			name: "#1_nil",
			gave: nil,
			want: []uint{0},
		},
		{
			name: "#2_[]",
			gave: []uint{},
			want: []uint{0},
		},
		{
			name: "#3_[0]",
			gave: []uint{0},
			want: []uint{0},
		},
		{
			name: "#4_[1_2_3]",
			gave: []uint{1, 2, 3},
			want: []uint{1, 2, 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got.SetHiddenLayer(tt.gave...); !reflect.DeepEqual(got.HiddenLayer, tt.want) {
				t.Errorf("SetHiddenLayer()\ngot:\t%v\nwant:\t%v", got.HiddenLayer, tt.want)
			}
		})
	}
}

func TestNN_GetActivationMode(t *testing.T) {
	want := &NN{ActivationMode: params.SIGMOID}
	t.Run("ModeSIGMOID", func(t *testing.T) {
		if got := want.GetActivationMode(); got != want.ActivationMode {
			t.Errorf("GetActivationMode() = %d, want %d", got, want.ActivationMode)
		}
	})
}

func TestNN_SetActivationMode(t *testing.T) {
	got := &NN{}
	want := params.LINEAR
	t.Run("ModeLINEAR", func(t *testing.T) {
		if got.SetActivationMode(want); got.ActivationMode != want {
			t.Errorf("SetActivationMode() = %d, want %d", got.ActivationMode, want)
		}
	})
}

func TestNN_GetLossMode(t *testing.T) {
	want := &NN{LossMode: params.ARCTAN}
	t.Run("ModeARCTAN", func(t *testing.T) {
		if got := want.GetLossMode(); got != want.LossMode {
			t.Errorf("GetLossMode() = %d, want %d", got, want.LossMode)
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
			if tt.got.SetLossMode(tt.gave); tt.got.LossMode != tt.want {
				t.Errorf("SetLossMode() = %d, want %d", tt.got.LossMode, tt.want)
			}
		})
	}
}

func TestNN_GetLossLimit(t *testing.T) {
	want := &NN{LossLimit: .1}
	t.Run("0.1", func(t *testing.T) {
		if got := want.GetLossLimit(); got != want.LossLimit {
			t.Errorf("GetLossLimit() = %f, want %f", got, want.LossLimit)
		}
	})
}

func TestNN_SetLossLimit(t *testing.T) {
	got := &NN{}
	want := .01
	t.Run("0.01", func(t *testing.T) {
		if got.SetLossLimit(want); got.LossLimit != want {
			t.Errorf("SetLossLimit() = %f, want %f", got.LossLimit, want)
		}
	})
}

func TestNN_GetRate(t *testing.T) {
	want := &NN{Rate: .3}
	t.Run("DefaultRate", func(t *testing.T) {
		if got := want.GetRate(); got != float64(want.Rate) {
			t.Errorf("GetRate() = %f, want %f", got, want.Rate)
		}
	})
}

func TestNN_SetRate(t *testing.T) {
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
			if tt.got.SetRate(tt.gave); tt.got.Rate != tt.want {
				t.Errorf("SetRate() = %f, want %f", tt.got.Rate, tt.want)
			}
		})
	}
}

func TestNN_GetWeight(t *testing.T) {
	tests := []struct {
		name string
		gave *NN
		want pkg.Float3Type
	}{
		{
			name: "#1_nil",
			gave: &NN{Weight: nil},
			want: nil,
		},
		{
			name: "#2_[]",
			gave: &NN{Weight: pkg.Float3Type{}},
			want: pkg.Float3Type{},
		},
		{
			name: "#3_[[[0.1_0.2_0.3]]]",
			gave: &NN{Weight: pkg.Float3Type{{{.1, .2, .3}}}},
			want: pkg.Float3Type{{{.1, .2, .3}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := *tt.gave.GetWeight().(*pkg.Float3Type); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetWeight()\ngot:\t%v\nwant:\t%v", got, tt.want)
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
			if got.SetWeight(tt.want); !reflect.DeepEqual(got.Weight, tt.want) {
				t.Errorf("SetWeight()\ngot:\t%v\nwant:\t%v", got.Weight, tt.want)
			}
		})
	}
}
