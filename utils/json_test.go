package utils

import (
	"path/filepath"
	"reflect"
	"testing"
)

var testJSON = filepath.Join("..", "testdata", "perceptron.json")

/*func TestFileJSON_Decode(t *testing.T) {
	type fields struct {
		Name string
	}
	type args struct {
		data interface{}
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j := &FileJSON{
				Name: tt.fields.Name,
			}
			if err := j.Decode(tt.args.data); (err != nil) != tt.wantErr {
				t.Errorf("Decode() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestFileJSON_Encode(t *testing.T) {
	type fields struct {
		Name string
	}
	type args struct {
		data interface{}
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j := &FileJSON{
				Name: tt.fields.Name,
			}
			if err := j.Encode(tt.args.data); (err != nil) != tt.wantErr {
				t.Errorf("Encode() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}*/

func TestFileJSON_GetValue(t *testing.T) {
	testFile := &FileJSON{Name: testJSON}
	tests := []struct {
		name string
		file *FileJSON
		gave string
		want interface{}
	}{
		{
			name: "#1_bias",
			file: testFile,
			gave: "bias",
			want: true,
		},
		{
			name: "#2_hidden",
			file: testFile,
			gave: "hidden",
			want: []interface{}{2.},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.file.GetValue(tt.gave); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetValue()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
	/*tests := []struct {
		name string
		key  string
		file JsonString
		want interface{}
	}{
		{
			name: "#1_name",
			key:  "name",
			file: JsonString(testNameJSON),
			want: perceptronName,
		},
		{
			name: "#2_bias",
			key:  "bias",
			file: JsonString(testNameJSON),
			want: true,
		},
		{
			name: "#3_hidden",
			key:  "hidden",
			file: JsonString(testNameJSON),
			want: []interface{}{2.},
		},
		{
			name: "#4_no_file",
			key:  "",
			file: JsonString(""),
			want: nil,
		},
		{
			name: "#5_not_read_file",
			key:  "",
			file: JsonString("perceptron"),
			want: nil,
		},
		{
			name: "#6_error_unmarshal",
			key:  "",
			file: JsonString("./json.go"),
			want: nil,
		},
		{
			name: "#7_warning_key",
			key:  "",
			file: JsonString(testNameJSON),
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.file.getValue(tt.key)
			if _, ok := got.(error); ok {
				got = nil
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getValue()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}*/
}
