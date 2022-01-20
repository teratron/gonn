package hopfield

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn/pkg"
)

func TestNN_GetEnergy(t *testing.T) {
	want := &NN{Energy: .015}
	t.Run("0.015", func(t *testing.T) {
		if got := want.GetEnergy(); got != want.Energy {
			t.Errorf("GetEnergy() = %f, want %f", got, want.Energy)
		}
	})
}

func TestNN_SetEnergy(t *testing.T) {
	got := &NN{}
	want := .015
	t.Run("0.015", func(t *testing.T) {
		if got.SetEnergy(want); got.Energy != want {
			t.Errorf("SetEnergy() = %f, want %f", got.Energy, want)
		}
	})
}

func TestNN_GetWeight(t *testing.T) {
	tests := []struct {
		name string
		gave *NN
		want pkg.Float2Type
	}{
		{
			name: "#1_nil",
			gave: &NN{Weight: nil},
			want: nil,
		},
		{
			name: "#2_[]",
			gave: &NN{Weight: pkg.Float2Type{}},
			want: pkg.Float2Type{},
		},
		{
			name: "#3_[[0.1_0.2_0.3]]",
			gave: &NN{Weight: pkg.Float2Type{{.1, .2, .3}}},
			want: pkg.Float2Type{{.1, .2, .3}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := *tt.gave.GetWeight().(*pkg.Float2Type); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetWeight()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func TestNN_SetWeight(t *testing.T) {
	got := &NN{}
	tests := []struct {
		name string
		want pkg.Float2Type
	}{
		{
			name: "#1_nil",
			want: nil,
		},
		{
			name: "#2_[]",
			want: pkg.Float2Type{},
		},
		{
			name: "#3_[[0.1_0.2_0.3]]",
			want: pkg.Float2Type{{.1, .2, .3}},
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
