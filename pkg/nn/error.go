package nn

import (
	"errors"
	"fmt"
	"log"
	"os"
)

var (
	ErrNotTrained = errors.New("network isn't trained")
)

type Response struct {
	Err struct {
		Id   int    `json:"id" xml:"id"`
		Text string `json:"text" xml:"text"`
	} `json:"error,omitempty"`
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
		log.Println("nn error:", err)
	}
	os.Exit(1)
}
