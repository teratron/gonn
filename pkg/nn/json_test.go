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
	want := "perceptron.json"
	t.Run(want, func(t *testing.T) {
		if got := jsonString(want).toString(); got != want {
			t.Errorf("toString() = %s, want %s", got, want)
		}
	})
}

func Test_jsonString_getValue(t *testing.T) {
	tests := []struct {
		name string
		key  string
		gave jsonString
		want interface{}
	}{
		{
			name: "#1_name",
			key:  "name",
			gave: jsonString("./testdata/perceptron.json"),
			want: "perceptron",
		},
		{
			name: "#2_bias",
			key:  "bias",
			gave: jsonString("./testdata/perceptron.json"),
			want: true,
		},
		{
			name: "#3_hidden",
			key:  "hidden",
			gave: jsonString("./testdata/perceptron.json"),
			want: []interface{}{5.},
		},
		{
			name: "#4_no_file",
			key:  "",
			gave: jsonString(""),
			want: nil,
		},
		{
			name: "#5_not_read_file",
			key:  "",
			gave: jsonString("perceptron"),
			want: nil,
		},
		{
			name: "#6_error_unmarshal",
			key:  "",
			gave: jsonString("./json.go"),
			want: nil,
		},
		{
			name: "#7_warning_key",
			key:  "",
			gave: jsonString("./testdata/perceptron.json"),
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.gave.getValue(tt.key)
			//fmt.Println(reflect.ValueOf(got).Type().Name())
			/*if g, ok := got.([]interface{}); ok {
				for _, i2 := range g {
					fmt.Printf("%T - %v\n", i2, i2)
				}
			}*/
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getValue() = %T - %v, want %T - %v", got, got, tt.want, tt.want)
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
