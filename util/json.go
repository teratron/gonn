package util

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/teratron/gonn"
)

type FileJSON struct {
	file string
}

// Decode
func (j *FileJSON) Decode(data interface{}) error {
	file, err := os.OpenFile(j.file, os.O_RDONLY, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		err = json.NewDecoder(file).Decode(data)
	}
	return err
}

// Encode
func (j *FileJSON) Encode(data interface{}) error {
	file, err := os.OpenFile(j.file, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		err = json.NewEncoder(file).Encode(data)
	}
	return err
}

type JsonString string

// JSON
func JSON(filename ...string) gonn.ReadWriter {
	if len(filename) > 0 {
		return JsonString(filename[0])
	}
	return JsonString("")
}

func (j JsonString) FileName() (filename string, err error) {
	filename = string(j)
	if len(filename) == 0 {
		err = gonn.ErrNoFile
	}
	return
}

func (j JsonString) GetValue(key string) interface{} {
	var (
		filename string
		err      error
		b        []byte
		data     interface{}
	)

	filename, err = j.FileName()
	if err == nil {
		b, err = ioutil.ReadFile(filename)
		if err == nil {
			err = json.Unmarshal(b, &data)
			if err == nil {
				if value, ok := data.(map[string]interface{})[key]; ok {
					return value
				}
				err = fmt.Errorf("key not found")
			}
		}
	}
	return fmt.Errorf("json get value: %w", err)
}

// Read
func (j JsonString) Read(reader gonn.Reader) (err error) {
	var (
		filename string
		b        []byte
	)

	filename, err = j.FileName()
	if err == nil {
		b, err = ioutil.ReadFile(filename)
		if err == nil {
			err = json.Unmarshal(b, &reader)
		}
	}

	if err != nil {
		err = fmt.Errorf("json read: %w", err)
	}
	return
}

var defaultNameJSON = filepath.Join("neural_network.json")

// Write
func (j JsonString) Write(writer ...gonn.Writer) (err error) {
	if len(writer) > 0 {
		if n, ok := writer[0].(gonn.NeuralNetwork); ok {
			var b []byte
			b, err = json.MarshalIndent(&n, "", "\t")
			if err == nil {
				filename := string(j)
				if len(filename) == 0 {
					if name := n.NameJSON(); len(name) > 0 {
						filename = name
					} else {
						filename = defaultNameJSON
					}
				}
				err = ioutil.WriteFile(filename, b, os.ModePerm)
			}
		}
	} else {
		err = fmt.Errorf("%w args", gonn.ErrEmpty)
	}

	if err != nil {
		err = fmt.Errorf("json write: %w", err)
	}
	return
}
