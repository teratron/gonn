package utils

import (
	"reflect"
	"testing"
)

func TestFileError_Error(t *testing.T) {
	type fields struct {
		Filer Filer
		Err   error
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
			f := &FileError{
				Filer: tt.fields.Filer,
				Err:   tt.fields.Err,
			}
			if got := f.Error(); got != tt.want {
				t.Errorf("Error() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetFileEncoding(t *testing.T) {
	type args struct {
		data []byte
	}
	tests := []struct {
		name string
		args args
		want Filer
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetFileEncoding(tt.args.data); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetFileEncoding() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetFileType(t *testing.T) {
	type args struct {
		name string
	}
	tests := []struct {
		name string
		args args
		want Filer
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetFileType(tt.args.name); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetFileType() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestReadFile(t *testing.T) {
	type args struct {
		data string
	}
	tests := []struct {
		name string
		args args
		want Filer
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ReadFile(tt.args.data); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ReadFile() = %v, want %v", got, tt.want)
			}
		})
	}
}
