package nn

import (
	"encoding/json"
	"errors"
	"log"
	"os"
)

var (
	ErrInit          = errors.New("initialization error")
	ErrNotRecognized = errors.New("not recognized")
	ErrMissingType   = errors.New("type is missing")
	ErrNoInput       = errors.New("no input data")
	ErrNoTarget      = errors.New("no target data")
	ErrEmpty         = errors.New("empty")
)

// LogError
func LogError(err error) {
	switch e := err.(type) {

	// OS
	case *os.LinkError:
		log.Println("link error:", e)
	case *os.PathError:
		log.Println("path error:", e)
	case *os.SyscallError:
		log.Println("syscall error:", e)

	// JSON
	case *json.SyntaxError:
		log.Println("syntax json error:", e, "offset:", e.Offset)
	case *json.UnmarshalTypeError:
		log.Println("unmarshal json error:", e, "offset:", e.Offset)
	case *json.MarshalerError:
		log.Println("marshaling json error:", e)

	default:
		log.Println(err)
	}
}
