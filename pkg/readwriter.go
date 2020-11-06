package pkg

// ReadWriter
type ReadWriter interface {
	Reader
	Writer
}

// Reader
type Reader interface {
	Read(Reader)
}

// Writer
type Writer interface {
	Write(...Writer)
}
