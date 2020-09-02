package nn

import (
	"encoding/xml"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type xmlType string

// XML
func XML(filename ...string) pkg.ReadWriter {
	return xmlType(filename[0])
}

// Read
func (j xmlType) Read(reader pkg.Reader) {
	if n, ok := reader.(*nn); ok {
		filename := string(j)
		if len(filename) == 0 {
			errXML(fmt.Errorf("file xml is missing\n"))
		}
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			errOS(err)
		}
		//fmt.Println(string(b))

		err = xml.Unmarshal(b, &n)
		if err != nil {
			errXML(err)
		}
		//fmt.Println(n)
		n.Architecture = nil
		n.IsInit = false
		n.xml = filename

		var data interface{}
		err = xml.Unmarshal(b, &data)
		if err != nil {
			errXML(err)
		}
		//fmt.Println(data)

		/*for key, value := range data.(map[string]interface{}) {
			if v, ok := value.(map[string]interface{}); ok {
				if key == "architecture" {
					b, err = xml.Marshal(&v)
					if err != nil {
						errXML(err)
					}

					err = xml.Unmarshal(b, &data)
					if err != nil {
						errXML(err)
					}

					for k, v := range data.(map[string]interface{}) {
						switch k {
						case "perceptron":
							n.Architecture = &perceptron{
								Architecture: n,
							}
							if a, ok := n.Architecture.(*perceptron); ok {
								a.readXML(v)
							}
						case "hopfield":
							n.Architecture = &hopfield{
								Architecture: n,
							}
							if a, ok := n.Architecture.(*hopfield); ok {
								a.readXML(v)
							}
						default:
							errNN(ErrNotRecognized)
						}
					}
				}
			}
		}*/
	}
}

// Write
func (j xmlType) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*nn); ok {
			filename := string(j)
			if len(filename) == 0 {
				if len(n.xml) > 0 {
					filename = n.xml
				} else {
					// TODO: generate filename
					filename = "config/neural_network.json"
				}
			}
			if n.IsTrain {
				n.Copy(Weight())
			} else {
				//log.Println("Not trained network")
				errNN(ErrNotTrained)
			}
			if b, err := xml.MarshalIndent(&n, "", "\t"); err != nil {
				errXML(err)
			} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				errOS(err)
			}
		}
	} else {
		//pkg.Log("Empty write", true) // !!!
		errNN(ErrEmptyWrite)
	}
}

// errXML
func errXML(err error) {
	switch e := err.(type) {
	case *xml.SyntaxError:
		log.Println("syntax xml error:", e, "line:", e.Line)
	case *xml.UnsupportedTypeError:
		log.Println("unsupported type xml error:", e)
	case *xml.TagPathError:
		log.Println("tag path xml error:", e)
	case *xml.UnmarshalError:
		log.Println("unmarshal xml error:", e)
	default:
		log.Println("xml error:", err)
	}
	os.Exit(1)
}
