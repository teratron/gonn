package nn

import (
	"fmt"
	"os"
	"reflect"
	"testing"
)

const testNameJSON = "./testdata/perceptron.json"

func init() {
	defaultNameJSON = "./testdata/tmp.json"
}

func TestJSON(t *testing.T) {
	tests := []struct {
		name string
		file []string
		want jsonString
	}{
		{
			name: "#1",
			file: []string{testNameJSON, ""},
			want: jsonString(testNameJSON),
		},
		{
			name: "#2",
			file: []string{},
			want: jsonString(""),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := JSON(tt.file...); got != tt.want {
				t.Errorf("JSON() = %s, want %s", got, tt.want)
			}
		})
	}
}

func Test_jsonString_fileName(t *testing.T) {
	tests := []struct {
		name      string
		file      jsonString
		wantName  string
		wantError error
	}{
		{
			name:      "#1",
			file:      jsonString(testNameJSON),
			wantName:  testNameJSON,
			wantError: nil,
		},
		{
			name:      "#2",
			file:      jsonString(""),
			wantName:  "",
			wantError: ErrNoFile,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotName, gotError := jsonString(tt.wantName).fileName()
			if gotName != tt.wantName {
				t.Errorf("fileName() = %s, want %s", gotName, tt.wantName)
			}
			if gotError != tt.wantError {
				t.Errorf("fileName() = %v, want %v", gotError, tt.wantError)
			}
		})
	}
}

func Test_jsonString_getValue(t *testing.T) {
	tests := []struct {
		name string
		key  string
		file jsonString
		want interface{}
	}{
		{
			name: "#1_name",
			key:  "name",
			file: jsonString(testNameJSON),
			want: perceptronName,
		},
		{
			name: "#2_bias",
			key:  "bias",
			file: jsonString(testNameJSON),
			want: true,
		},
		{
			name: "#3_hidden",
			key:  "hidden",
			file: jsonString(testNameJSON),
			want: []interface{}{2.},
		},
		{
			name: "#4_no_file",
			key:  "",
			file: jsonString(""),
			want: nil,
		},
		{
			name: "#5_not_read_file",
			key:  "",
			file: jsonString("perceptron"),
			want: nil,
		},
		{
			name: "#6_error_unmarshal",
			key:  "",
			file: jsonString("./json.go"),
			want: nil,
		},
		{
			name: "#7_warning_key",
			key:  "",
			file: jsonString(testNameJSON),
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
	}
}

func Test_jsonString_Read(t *testing.T) {
	existErr := fmt.Errorf("error")
	tests := []struct {
		name    string
		file    jsonString
		got     Reader
		want    Reader
		wantErr error
	}{
		{
			name: "#1_perceptron",
			file: jsonString(testNameJSON),
			got:  &perceptron{},
			want: &perceptron{
				Name:       perceptronName,
				Bias:       true,
				Hidden:     []int{2},
				Activation: ModeSIGMOID,
				Loss:       ModeMSE,
				Limit:      .1,
				Rate:       DefaultRate,
				Weights: Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
			},
			wantErr: nil,
		},
		{
			name:    "#2_no_file",
			file:    jsonString(""),
			want:    nil,
			wantErr: existErr,
		},
		{
			name:    "#3_not_read_file",
			file:    jsonString("perceptron"),
			want:    nil,
			wantErr: existErr,
		},
		{
			name:    "#4_error_unmarshal",
			file:    jsonString("./json.go"),
			want:    nil,
			wantErr: existErr,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErr := tt.file.Read(tt.got)
			if (gotErr != nil && tt.wantErr == nil) || (gotErr == nil && tt.wantErr != nil) {
				t.Errorf("Read()\ngot error:\t%v\nwant error:\t%v", gotErr, tt.wantErr)
			}
			if !reflect.DeepEqual(tt.got, tt.want) {
				t.Errorf("Read()\ngot:\t%v\nwant:\t%v", tt.got, tt.want)
			}
		})
	}
}

func Test_jsonString_Write(t *testing.T) {
	tests := []struct {
		name    string
		file    jsonString
		got     *perceptron
		want    []Writer
		wantErr error
	}{
		{
			name:    "#1_perceptron",
			file:    jsonString(defaultNameJSON),
			got:     &perceptron{},
			want:    []Writer{&perceptron{}},
			wantErr: nil,
		},
		{
			name:    "#2_no_args",
			file:    jsonString(""),
			want:    []Writer{},
			wantErr: fmt.Errorf("error"),
		},
		{
			name: "#3_no_filename",
			file: jsonString(""),
			got: &perceptron{
				jsonName: defaultNameJSON,
			},
			want:    []Writer{&perceptron{}},
			wantErr: nil,
		},
		{
			name:    "#4_no_filename",
			file:    jsonString(""),
			got:     &perceptron{},
			want:    []Writer{&perceptron{}},
			wantErr: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.file) == 0 && len(tt.want) > 0 && len(tt.got.jsonName) > 0 {
				tt.want[0].(*perceptron).jsonName = defaultNameJSON
			}
			gotErr := tt.file.Write(tt.want...)
			if (tt.wantErr == nil && gotErr != nil) || (tt.wantErr != nil && gotErr == nil) {
				t.Errorf("Write()\ngot error:\t%v\nwant error:\t%v", gotErr, tt.wantErr)
			}
			if len(tt.want) > 0 {
				defer func() {
					if err := os.Remove(string(tt.file)); err != nil {
						t.Error(err)
					}
				}()
				if len(tt.file) == 0 {
					tt.file = jsonString(defaultNameJSON)
				}
				_ = tt.file.Read(tt.got)
				if !reflect.DeepEqual(tt.got, tt.want[0]) {
					t.Errorf("Write()\ngot:\t%v\nwant:\t%v", tt.got, tt.want[0])
				}
			}
		})
	}
}
