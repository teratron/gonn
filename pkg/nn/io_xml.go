package nn

import (
	"encoding/xml"
	"io/ioutil"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type xmlType string

// XML
func XML(filename ...string) pkg.ReaderWriter {
	return xmlType(filename[0])
}

func (j xmlType) Read(reader pkg.Reader) {
	if r, ok := reader.(pkg.Reader); ok {
		r.Read(j)
	}
}

func (j xmlType) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if w, ok := writer[0].(pkg.Writer); ok {
			w.Write(j)
		}
	} else {
		pkg.Log("Empty write", true) // !!!
	}
}

func (n *NN) readXML(value interface{}) {
	filename, ok := value.(string)
	if ok {
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			log.Fatal("Can't load xml: ", err)
		}
		//fmt.Println(string(b))

		err = xml.Unmarshal(b, &n)
		if err != nil {
			log.Println(err)
		}
		//fmt.Println(n)
		n.Architecture = nil
		n.IsInit       = false
		n.xml          = filename

		var data interface{}
		err = xml.Unmarshal(b, &data)
		if err != nil {
			log.Println(err)
		}
		//fmt.Println(data)

		/*for key, value := range data.(map[string]interface{}) {
			if v, ok := value.(map[string]interface{}); ok {
				if key == "architecture" {
					b, err = xml.Marshal(&v)
					if err != nil {
						log.Println(err)
					}

					err = xml.Unmarshal(b, &data)
					if err != nil {
						log.Println(err)
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
						}
					}
				}
			}
		}*/
	}
}

func (n *NN) writeXML(filename string) {
	if n.IsTrain {
		n.Copy(Weight())
	} else {
		log.Println("Not trained network")
	}
	if b, err := xml.MarshalIndent(&n, "", "\t"); err != nil {
		log.Fatal("XML marshaling failed: ", err)
	} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
		log.Fatal("Can't write file:", err)
	}
}