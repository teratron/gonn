package layers

import (
	"reflect"
	"testing"
)

func TestCheckLayers(t *testing.T) {
	type args struct {
		layers []uint
	}
	tests := []struct {
		name string
		args
		want []uint
	}{
		{
			name: "#1_normal",
			args: args{[]uint{1, 2, 3}},
			want: []uint{1, 2, 3},
		}, {
			name: "#2_length_0",
			args: args{[]uint{}},
			want: []uint{0},
		}, {
			name: "#3_nil",
			args: args{nil},
			want: []uint{0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CheckLayers(tt.args.layers); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CheckLayers() = %v, want %v", got, tt.want)
			}
		})
	}
}
