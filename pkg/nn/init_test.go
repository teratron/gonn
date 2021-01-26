package nn

import (
	"reflect"
	"strconv"
	"testing"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name   string
		reader []Reader
		want   NeuralNetwork
	}{
		{
			name:   "#1_warning",
			reader: []Reader{nil},
			want:   nil,
		},
		{
			name:   "#2_default_" + perceptronName,
			reader: nil,
			want:   Perceptron(),
		},
		{
			name:   "#3_default_" + perceptronName,
			reader: []Reader{},
			want:   Perceptron(),
		},
		{
			name:   "#4_" + perceptronName,
			reader: []Reader{Perceptron()},
			want:   Perceptron(),
		},
		{
			name:   "#5_" + hopfieldName,
			reader: []Reader{Hopfield()},
			want:   Hopfield(),
		},
		/*{
			name:   "#6_JSON",
			reader: []Reader{JSON("tmp.json")},
			want:   Hopfield(),
		},*/
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := New(tt.reader...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func Test_getArchitecture(t *testing.T) {
	tests := []struct {
		name string
		want NeuralNetwork
	}{
		{
			name: "warning",
			want: nil,
		},
		{
			name: perceptronName,
			want: Perceptron(),
		},
		{
			name: hopfieldName,
			want: Hopfield(),
		},
	}
	for i, tt := range tests {
		t.Run("#"+strconv.Itoa(i+1)+"_"+tt.name, func(t *testing.T) {
			if got := getArchitecture(tt.name); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getArchitecture(%s)\ngot:\t%v\nwant:\t%v", tt.name, got, tt.want)
			}
		})
	}
}

func Test_getMaxIteration(t *testing.T) {
	want := MaxIteration
	t.Run("MaxIteration", func(t *testing.T) {
		if got := getMaxIteration(); got != want {
			t.Errorf("getMaxIteration() = %d, want %d", got, want)
		}
	})
}

func Test_getRandFloat(t *testing.T) {
	want := [3]float64{-.5, 0, .5}
	for i := range want {
		t.Run("#"+strconv.Itoa(i+1), func(t *testing.T) {
			if got := getRandFloat(); got < want[0] || got == want[1] || got > want[2] {
				t.Errorf("getRandFloat() = %.3f", got)
			}
		})
	}
}
