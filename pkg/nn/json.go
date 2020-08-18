package nn

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type jsonType string

//
func JSON(filename ...string) pkg.ReaderWriter {
	/*if len(filename) > 0 && filename[0] != "" {
		return jsonType(filename[0])
	} else {
		return jsonType("")
	}*/
	return jsonType(filename[0])
}

/*func (j jsonType) Read([]byte) (n int, err error) {
	return
}

func (j jsonType) Write([]byte) (n int, err error) {
	return
}*/

func (j jsonType) Read(pkg.Reader) {}
func (j jsonType) Write(...pkg.Writer) {}

func (n *NN) readJSON(filename string) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load json: ", err)
	}
	//fmt.Println(string(b))

	// Декодируем json в NN
	err = json.Unmarshal(b, &n)
	if err != nil {
		log.Println(err)
	}
	//fmt.Println(n)
	//fmt.Println(n.Architecture)
	//n.Architecture = nil
	n.json = filename

	// Декодируем json в тип map[string]interface{}
	var data interface{}
	err = json.Unmarshal(b, &data)
	if err != nil {
		log.Println(err)
	}
	//fmt.Println(data)

	for key, value := range data.(map[string]interface{}) {
		if v, ok := value.(map[string]interface{}); ok {
			if key == "architecture" {
				b, err = json.Marshal(&v)
				if err != nil {
					log.Println(err)
				}
				//fmt.Println(string(b))

				err = json.Unmarshal(b, &data)
				if err != nil {
					log.Println(err)
				}
				//fmt.Println(data)

				for k, v := range data.(map[string]interface{}) {
					//fmt.Printf("%s - %T - %v\n", k, v, v)
					switch k {
					case "perceptron":
						n.Architecture = &perceptron{}
						if a, ok := n.Architecture.(*perceptron); ok {
							//a.Configuration.Bias = true
							//a.Configuration.Rate = 0.89
							//fmt.Println(a.Configuration)
							a.readJSON(v)
						}
					case "hopfield":
						n.Architecture = &hopfield{}
						if a, ok := n.Architecture.(*hopfield); ok {
							a.readJSON(v)
						}
					default:
					}
				}
			}
		}
	}
	//fmt.Println("+++++++++", n)
}

func (n *NN) writeJSON(filename string) {
	//n.Architecture.(Architecture).getWeight()
	if n.IsTrain {
		n.Get().Get(Weight())
	} else {
		log.Println("Not trained network")
	}
	if b, err := json.MarshalIndent(&n, "", "\t"); err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
		log.Fatal("Can't write file:", err)
	}
}

/*func (p *perceptron) getWeight() {
	fmt.Println("func (p *perceptron) getWeight() ")
}*/