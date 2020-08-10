//
package pkg

type ReaderWriter interface {
	Reader
	Writer
}

type Reader interface {
	Read(Reader)

}

type Writer interface {
	Write(...Writer)
}