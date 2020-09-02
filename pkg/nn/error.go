package nn

import (
	"errors"
	"fmt"
	"log"
)

var (
	ErrNotTrained    = errors.New("network is not trained")
	ErrNotRecognized = errors.New("network is not recognized")
	ErrMissingType   = errors.New("type is missing")
	ErrNoTarget      = errors.New("no target data")
)

type Response struct {
	Err struct {
		Id   int    `json:"id" xml:"id"`
		Text string `json:"text" xml:"text"`
	} `json:"error,omitempty"`
}

type nnError struct {
	Err     error
	Message string
	Code    int
}

/*func myErrorB() error {
	resp := new(Response)
	resp.Err.Id = 10
	resp.Err.Text = "resp error"
	return resp
}*/

func (r *Response) Error() string {
	return fmt.Sprintf("%d: %s\n", r.Err.Id, r.Err.Text)
}

func (e *nnError) Error() string {
	return fmt.Sprintf("%d: %s %v\n", e.Code, e.Message, e.Err)
}

// errNN
func errNN(err error) {
	switch e := err.(type) {
	case error:
		log.Println("syntax json error:", e)
	/*case :
		log.Println("unmarshal json error:", e)
	case :
		log.Println("marshaling json error:", e)*/
	default:
		log.Println("error:", err)
	}
}
