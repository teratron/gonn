package nn

// Controller
type Controller interface {
	GetSetter
	ReadWriter
}

// GetSetter
type GetSetter interface {
	Getter
	Setter
}

// Getter
type Getter interface {
	Get(...Getter) GetSetter
}

// Setter
type Setter interface {
	Set(...Setter)
}

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
