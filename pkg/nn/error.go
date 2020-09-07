package nn

import (
	"errors"
	"log"
)

var (
	//ErrNotFound      = errors.New("not found")
	ErrInit          = errors.New("initialization error")
	ErrNotTrained    = errors.New("network is not trained")
	ErrNotRecognized = errors.New("network is not recognized")
	ErrMissingType   = errors.New("type is missing")
	ErrNoTarget      = errors.New("no target data")
	ErrEmpty         = errors.New("empty")
)

// errNN
func errNN(err error) {
	switch e := err.(type) {
	case error:
		log.Println("error:", e)
	/*case :
		log.Println("unmarshal json error:", e)
	case :
		log.Println("marshaling json error:", e)*/
	default:
		log.Println("error:", err)
	}
}