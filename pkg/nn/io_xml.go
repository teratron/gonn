package nn

import (
	"encoding/xml"
	"github.com/zigenzoog/gonn/pkg"
	"io/ioutil"
	"log"
	"os"
)

type xmlType string

// XML
func XML(filename ...string) pkg.ReadWriter {
	return xmlType(filename[0])
}

// Read
func (j xmlType) Read(reader pkg.Reader) {
	if n, ok := reader.(*NN); ok {
		filename := string(j)
		if len(filename) == 0 {
			log.Fatal("Отсутствует название файла нейросети для XML")
		}
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

// Write
func (j xmlType) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*NN); ok {
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
				log.Println("Not trained network")
			}
			if b, err := xml.MarshalIndent(&n, "", "\t"); err != nil {
				log.Fatal("XML marshaling failed: ", err)
			} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				log.Fatal("Can't write file:", err)
			}
		}
	} else {
		pkg.Log("Empty write", true) // !!!
	}
}