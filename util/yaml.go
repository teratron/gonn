package util

import (
	"os"

	"gopkg.in/yaml.v2"
)

type FileYAML struct {
	file string
}

// Decode
func (y *FileYAML) Decode(data interface{}) error {
	file, err := os.OpenFile(y.file, os.O_RDONLY, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		err = yaml.NewDecoder(file).Decode(data)
	}
	return err
}

// Encode
func (y *FileYAML) Encode(data interface{}) error {
	file, err := os.OpenFile(y.file, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		err = yaml.NewEncoder(file).Encode(data)
	}
	return err
}
