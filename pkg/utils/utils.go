package utils

import (
	"encoding/json"
	"fmt"
	"path/filepath"
)

// Filer.
type Filer interface {
	Decode(interface{}) error
	Encode(interface{}) error
	GetValue(key string) interface{}
	GetName() string
	ClearData()
}

// FileError.
type FileError struct {
	Filer
	Err error
}

// Error.
func (f *FileError) Error() string {
	return fmt.Sprintf("file type error: %v\n", f.Err)
}

// ReadFile.
func ReadFile(data string) Filer {
	f := GetFileEncoding([]byte(data))
	if err, ok := f.(*FileError); ok {
		f = GetFileType(data)
		if _, ok = f.(*FileError); ok {
			return &FileError{Err: fmt.Errorf("utils.ReadFile: %v and %v", err, f)}
		}
	}

	return f
}

// GetFileType.
func GetFileType(name string) Filer {
	ext := filepath.Ext(name)
	switch ext {
	case ".json":
		return &FileJSON{Name: name}
	}

	return &FileError{Err: fmt.Errorf("utils.GetFileType extension isn't defined")}
}

// GetFileEncoding.
func GetFileEncoding(data []byte) Filer {
	switch {
	case json.Valid(data):
		return &FileJSON{Data: data}
	}

	return &FileError{Err: fmt.Errorf("utils.GetFileEncoding invalid encoding data")}
}
