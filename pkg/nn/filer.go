package nn

// Filer
type Filer interface {
	ReadWriter
	getValue(key string) interface{}
	fileName() (string, error)
}
