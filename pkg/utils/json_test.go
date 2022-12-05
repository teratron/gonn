package utils

import (
	"path/filepath"
	"reflect"
	"testing"
)

var testJSON = filepath.Join("..", "testdata", "perceptron.json")

func TestFileJSON_Decode(t *testing.T) {
	type fields struct {
		Name string
		Data []byte
	}
	type args struct {
		value interface{}
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
				Data: tt.fields.Data,
			}
			if err := j.Decode(tt.args.value); (err != nil) != tt.wantErr {
				t.Errorf("Decode() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestFileJSON_Encode(t *testing.T) {
	type fields struct {
		Name string
		Data []byte
	}
	type args struct {
		value interface{}
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
				Data: tt.fields.Data,
			}
			if err := j.Encode(tt.args.value); (err != nil) != tt.wantErr {
				t.Errorf("Encode() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

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
		{
			name: "#7_error_key",
			file: testFile,
			gave: "error",
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.file.GetValue(tt.gave)
			if _, ok := got.(error); ok {
				got = nil
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetValue()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}

func TestFileJSON_GetName(t *testing.T) {
	type fields struct {
		Name string
		Data []byte
	}
	tests := []struct {
		name   string
		fields fields
		want   string
	}{
		// TODO: Add test cases.
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j := &FileJSON{
				Name: tt.fields.Name,
				Data: tt.fields.Data,
			}
			if got := j.GetName(); got != tt.want {
				t.Errorf("GetName() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFileJSON_ClearData(t *testing.T) {
	type fields struct {
		Name string
		Data []byte
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j := &FileJSON{
				Name: tt.fields.Name,
				Data: tt.fields.Data,
			}
			j.ClearData()
		})
	}
}
