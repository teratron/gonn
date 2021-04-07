package hopfield

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn"
)

func TestNN_NeuronEnergy(t *testing.T) {
	want := &NN{Energy: .015}
	t.Run("0.015", func(t *testing.T) {
		if got := want.NeuronEnergy(); got != want.Energy {
			t.Errorf("NeuronEnergy() = %f, want %f", got, want.Energy)
		}
	})
}

func TestNN_SetNeuronEnergy(t *testing.T) {
	got := &NN{}
	want := .015
	t.Run("0.015", func(t *testing.T) {
		if got.SetNeuronEnergy(want); got.Energy != want {
			t.Errorf("SetNeuronEnergy() = %f, want %f", got.Energy, want)
		}
	})
}

func TestNN_Weight(t *testing.T) {
	tests := []struct {
		name string
		gave *NN
		want gonn.Float2Type
	}{
		{
			name: "#1_nil",
			gave: &NN{Weights: nil},
			want: nil,
		},
		{
			name: "#2_[]",
			gave: &NN{Weights: gonn.Float2Type{}},
			want: gonn.Float2Type{},
		},
		{
			name: "#3_[[0.1_0.2_0.3]]",
			gave: &NN{Weights: gonn.Float2Type{{.1, .2, .3}}},
			want: gonn.Float2Type{{.1, .2, .3}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := *tt.gave.Weight().(*gonn.Float2Type); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Weight()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func TestNN_SetWeight(t *testing.T) {
	got := &NN{}
	tests := []struct {
		name string
		want gonn.Float2Type
	}{
		{
			name: "#1_nil",
			want: nil,
		},
		{
			name: "#2_[]",
			want: gonn.Float2Type{},
		},
		{
			name: "#3_[[0.1_0.2_0.3]]",
			want: gonn.Float2Type{{.1, .2, .3}},
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
