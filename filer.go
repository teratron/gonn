package gonn

// Filer
type Filer interface {
	ReadWriter
	GetValue(key string) interface{}
	FileName() (string, error)
}
