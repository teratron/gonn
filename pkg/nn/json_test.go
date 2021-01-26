package nn

import (
	"reflect"
	"testing"
)

func TestJSON(t *testing.T) {
	tests := []struct {
		name     string
		filename []string
		want     jsonString
	}{
		{
			name:     "#1",
			filename: []string{"perceptron.json"},
			want:     jsonString("perceptron.json"),
		},
		{
			name:     "#2",
			filename: []string{"perceptron.json", ""},
			want:     jsonString("perceptron.json"),
		},
		{
			name:     "#3",
			filename: []string{},
			want:     jsonString(""),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := JSON(tt.filename...); got != tt.want {
				t.Errorf("JSON() = %s, want %s", got, tt.want)
			}
		})
	}
}

func Test_jsonString_toString(t *testing.T) {
	gave := jsonString("perceptron.json")
	want := "perceptron.json"
	t.Run(want, func(t *testing.T) {
		if got := gave.toString(); got != want {
			t.Errorf("toString() = %s, want %s", got, want)
		}
	})
}

func Test_jsonString_getValue(t *testing.T) {
	type args struct {
		key string
	}
	tests := []struct {
		name string
		j    jsonString
		args args
		want interface{}
	}{
		// TODO: Add test cases.
		{},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.j.getValue(tt.args.key); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getValue() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_jsonString_Read(t *testing.T) {
	type args struct {
		reader Reader
	}
	tests := []struct {
		name string
		j    jsonString
		args args
	}{
		// TODO: Add test cases.
		{},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
		})
	}
}

func Test_jsonString_Write(t *testing.T) {
	type args struct {
		writer []Writer
	}
	tests := []struct {
		name string
		j    jsonString
		args args
	}{
		// TODO: Add test cases.
		{},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
		})
	}
}
