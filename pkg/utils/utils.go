package utils

import (
	"fmt"
	"path/filepath"
)

// Filer.
type Filer interface {
	Decode(interface{}) error
	Encode(interface{}) error
	GetValue(key string) interface{}
	GetName() string
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

// GetFileType.
func GetFileType(name string) Filer {
	ext := filepath.Ext(name)
	switch ext {
	case ".json":
		return &FileJSON{name}
	case ".yml", ".yaml":
		return &FileYAML{name}
	default:
		return &FileError{Err: fmt.Errorf("extension isn't defined: %s", ext)}
	}
}
