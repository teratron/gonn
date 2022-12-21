package utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type FileJSON struct {
	Name string
	Data []byte
}

// Decode.
func (j *FileJSON) Decode(value interface{}) (err error) {
	var r io.Reader
	if j.Data != nil {
		r = bytes.NewReader(j.Data)
		j.Data = nil
	} else if len(j.Name) > 0 {
		r, err = os.OpenFile(j.Name, os.O_RDONLY, 0600)
		if err == nil {
			defer func() { err = r.(*os.File).Close() }()
		}
	}

	if r != nil {
		err = json.NewDecoder(r).Decode(value)
	}

	if err != nil {
		err = fmt.Errorf("utils.FileJSON.Decode: %v", err)
	}
	return
}

// Encode.
func (j *FileJSON) Encode(value interface{}) error {
	file, err := os.OpenFile(j.Name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		e := json.NewEncoder(file)
		e.SetIndent("", "\t")
		err = e.Encode(value)
	}

	if err != nil {
		err = fmt.Errorf("utils.FileJSON.Encode: %v", err)
	}
	return err
}

// GetValue returns value by key.
func (j *FileJSON) GetValue(key string) interface{} {
	var value interface{}
	var data []byte
	var err error

	if j.Data != nil {
		data = j.Data
	} else if len(j.Name) > 0 {
		data, err = os.ReadFile(j.Name)
		if err != nil {
			j.Name = ""
			goto ERROR
		}
	}

	err = json.Unmarshal(data, &value)
	if err == nil {
		if k, ok := value.(map[string]interface{})[key]; ok {
			return k
		}
		err = fmt.Errorf("key not found")
	}

ERROR:
	return fmt.Errorf("utils.FileJSON.GetValue: %v", err)
}

// GetName returns filename.
func (j *FileJSON) GetName() string {
	return j.Name
}

// ClearData clears data.
func (j *FileJSON) ClearData() {
	if j.Data != nil {
		j.Data = nil
	}
}
