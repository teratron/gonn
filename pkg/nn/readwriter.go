package nn

// ReadWriter
type ReadWriter interface {
	Reader
	Writer
}

// Reader
type Reader interface {
	Read(Reader) error
}

// Writer
type Writer interface {
	Write(...Writer) error
}
