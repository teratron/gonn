package util

import (
	"fmt"
	"path/filepath"
)

type DecodeEncoder interface {
	Decoder
	Encoder
}

type Decoder interface {
	Decode(interface{}) error
}

type Encoder interface {
	Encode(interface{}) error
}

type FileError struct {
	DecodeEncoder
	Err error
}

func (f *FileError) Error() string {
	return fmt.Sprintf("file type error: %v\n", f.Err)
}

func GetFileType(file string) DecodeEncoder {
	ext := filepath.Base(filepath.Ext(file))
	switch ext {
	case ".json":
		return &FileJSON{file}
	case ".yml":
		return &FileYAML{file}
	default:
		return &FileError{Err: fmt.Errorf("extension isn't defined: %s", ext)}
	}
}
