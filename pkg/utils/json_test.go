package utils

import (
	"path/filepath"
	"reflect"
	"testing"
)

var testJSON = filepath.Join("..", "testdata", "perceptron.json")

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
