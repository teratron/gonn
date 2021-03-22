package util

import (
	"fmt"
	"path/filepath"

	"github.com/teratron/gonn"
)

type FileError struct {
	gonn.Filer
	Err error
}

func (f *FileError) Error() string {
	return fmt.Sprintf("file type error: %v\n", f.Err)
}

func GetFileType(name string) gonn.Filer {
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
