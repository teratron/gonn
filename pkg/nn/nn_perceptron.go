package nn

import (
	"bytes"
	"fmt"
	"math"

	"github.com/teratron/gonn/pkg"
)

const perceptronName = "perceptron"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*perceptron)(nil)

// perceptron
type perceptron struct {
	Parameter `json:"-" xml:"-"`

	// Neural network architecture name
	Name string `json:"name" xml:"name"`

	// Array of the number of neurons in each hidden layer
	Hidden HiddenArrUint `json:"hidden" xml:"hidden>layer"`

	// The neuron bias, false or true
	Bias biasBool `json:"bias" xml:"bias"`

	// Activation function mode
	Activation uint8 `json:"activation" xml:"activation"`

	// The mode of calculation of the total error
	Loss uint8 `json:"loss" xml:"loss"`

	// Minimum (sufficient) limit of the average of the error during training
	Limit float64 `json:"limit" xml:"limit"`

	// Learning coefficient, from 0 to 1
	Rate FloatType `json:"rate" xml:"rate"`

	// Matrix
	Weights [][][]FloatType `json:"weights" xml:"weights>weights"` // Weight value
	neuron  [][]FloatType   // Neuron value
	miss    [][]FloatType   // Neuron error (miss)
	//input   []float64
	//target  []float64

	lastIndexLayer int
	lenInput       int
	lenOutput      int

	// State of the neural network
	isInit  bool
	isTrain bool

	// Config
	jsonName string
}

// Perceptron return perceptron neural network
func Perceptron() *perceptron {
	return &perceptron{
		Name:       perceptronName,
		Hidden:     HiddenArrUint{},
		Bias:       false,
		Activation: ModeSIGMOID,
		Loss:       ModeMSE,
		Limit:      .01,
		Rate:       FloatType(DefaultRate),
	}
}

func (p *perceptron) name() string {
	return p.Name
}

func (p *perceptron) setName(name string) {
	p.Name = name
}

func (p *perceptron) stateInit() bool {
	return p.isInit
}

func (p *perceptron) setStateInit(state bool) {
	p.isInit = state
}

func (p *perceptron) stateTrain() bool {
	return p.isTrain
}

func (p *perceptron) setStateTrain(state bool) {
	p.isTrain = state
}

func (p *perceptron) nameJSON() string {
	return p.jsonName
}

func (p *perceptron) setNameJSON(name string) {
	p.jsonName = name
}

// HiddenLayer
func (p *perceptron) HiddenLayer() []uint {
	return p.Hidden
}

// NeuronBias
func (p *perceptron) NeuronBias() bool {
	return bool(p.Bias)
}

// ActivationMode
func (p *perceptron) ActivationMode() uint8 {
	return p.Activation
}

// LossMode
func (p *perceptron) LossMode() uint8 {
	return p.Loss
}

// LossLimit
func (p *perceptron) LossLimit() float64 {
	return p.Limit
}

// LearningRate
func (p *perceptron) LearningRate() float32 {
	return float32(p.Rate)
}

// Weight
func (p *perceptron) Weight() Floater {
	return nil //p.getWeight()
}

// Set
func (p *perceptron) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		for _, a := range args {
			switch v := a.(type) {
			case HiddenArrUint:
				p.Hidden = v
			case biasBool:
				p.Bias = v
			case activationModeUint:
				p.Activation = uint8(v)
			case lossModeUint:
				p.Loss = uint8(v)
			case lossLimitFloat:
				p.Limit = float64(v)
			case rateFloat:
				p.Rate = FloatType(v)
			case *weight:
				//p.setWeight(v.buffer.(*Float3Type))
			default:
				pkg.LogError(fmt.Errorf("%T %w for perceptron", v, pkg.ErrMissingType))
			}
		}
	} else {
		pkg.LogError(fmt.Errorf("%w set for perceptron", pkg.ErrEmpty))
	}
}

// Get
func (p *perceptron) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		for _, a := range args {
			switch a.(type) {
			case HiddenArrUint:
				return p.Hidden
			case biasBool:
				return p.Bias
			case activationModeUint:
				return activationModeUint(p.Activation)
			case lossModeUint:
				return lossModeUint(p.Loss)
			case lossLimitFloat:
				return lossLimitFloat(p.Limit)
			case rateFloat:
				return p.Rate
			case *weight:
				//return p.getWeight()
			default:
				pkg.LogError(fmt.Errorf("%T %w for perceptron", a, pkg.ErrMissingType))
			}
		}
	}
	return p
}

// Copy
func (p *perceptron) Copy(copier pkg.Copier) {
	switch c := copier.(type) {
	case *weight:
		//p.copyWeight()
	default:
		pkg.LogError(fmt.Errorf("%T %w for copy: %v", c, pkg.ErrMissingType, c))
	}
}

// Paste
func (p *perceptron) Paste(paster pkg.Paster) {
	switch v := paster.(type) {
	case *weight:
		/*if err := p.pasteWeight(); err != nil {
			pkg.LogError(err)
		}*/
	default:
		pkg.LogError(fmt.Errorf("%T %w for paste: %v", v, pkg.ErrMissingType, v))
	}
}

// Read
func (p *perceptron) Read(reader pkg.Reader) {
	switch r := reader.(type) {
	case pkg.Filer:
		r.Read(p)
	default:
		pkg.LogError(fmt.Errorf("%T %w for read: %v", r, pkg.ErrMissingType, r))
	}
}

// Write
func (p *perceptron) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		for _, w := range writer {
			switch v := w.(type) {
			case pkg.Filer:
				v.Write(p)
			case *report:
				p.writeReport(v)
			default:
				pkg.LogError(fmt.Errorf("%T %w for write: %v", v, pkg.ErrMissingType, w))
			}
		}
	} else {
		pkg.LogError(fmt.Errorf("%w for write", pkg.ErrEmpty))
	}
}

// init initialize
func (p *perceptron) init(lenInput int, lenTarget ...interface{}) bool {
	if len(lenTarget) > 0 {
		var tmp HiddenArrUint
		defer func() {
			tmp = nil
		}()

		p.lastIndexLayer = len(p.Hidden)
		p.lenInput = lenInput
		p.lenOutput = lenTarget[0].(int)
		tmp = append(p.Hidden, uint(p.lenOutput))
		layer := make(HiddenArrUint, p.lastIndexLayer+1)
		lenLayer := copy(layer, tmp)

		bias := 0
		if p.Bias {
			bias = 1
		}

		p.Weights = make([][][]FloatType, lenLayer)
		p.neuron = make([][]FloatType, lenLayer)
		p.miss = make([][]FloatType, lenLayer)
		//p.input = make([]float64, p.lenInput)
		//p.target = make([]float64, p.lenOutput)

		//p.neuron = make([][]*neuron, lenLayer)
		//p.axon = make([][][]*axon, lenLayer)
		for i, l := range layer {
			p.Weights[i] = make([][]FloatType, lenLayer)
			p.neuron[i] = make([]FloatType, lenLayer)
			p.miss[i] = make([]FloatType, lenLayer)

			//p.neuron[i] = make([]*neuron, l)
			//p.axon[i] = make([][]*axon, l)
			for j := 0; j < int(l); j++ {
				if i == 0 {
					p.Weights[i][j] = make([]FloatType, p.lenInput+bias)
					//p.axon[i][j] = make([]*axon, p.lenInput+bias)
				} else {
					p.Weights[i][j] = make([]FloatType, int(layer[i-1])+bias)
					//p.axon[i][j] = make([]*axon, int(layer[i-1])+bias)
				}
				for k, w := range p.Weights[i][j] {
					fmt.Println(w, p.Weights[i][j][k])
					p.Weights[i][j][k] = getRand()
				}
			}
		}
		//p.initNeuron()
		//p.initAxon()

		return true
	}
	pkg.LogError(pkg.ErrNoTarget)
	return false
}

// reInit
func (p *perceptron) reInit() {
	bias := 0
	if p.Bias {
		bias = 1
	}
	length := len(p.Weights) - 1
	p.Hidden = make(HiddenArrUint, length)
	for i := range p.Hidden {
		p.Hidden[i] = uint(len(p.Weights[i]))
	}
	p.isInit = p.init(len(p.Weights[0][0])-bias, len(p.Weights[length]))
}

// initNeuron
/*func (p *perceptron) initNeuron() {
	for i, v := range p.neuron {
		for j := range v {
			p.neuron[i][j] = &neuron{
				axon: p.axon[i][j],
				miss: FloatType(0),
			}
		}
	}
}*/

// initAxon
/*func (p *perceptron) initAxon() {
	isTrain := true
	if !p.isTrain {
		isTrain = false
	}
	for i, v := range p.axon {
		for j, w := range v {
			for k := range w {
				p.axon[i][j][k] = &axon{
					synapse: map[string]Synapser{},
				}
				if !isTrain {
					//p.axon[i][j][k].weight = getRand()
					p.Weights[i][j][k] = getRand()
				}
				if i == 0 {
					if k < p.lenInput {
						p.axon[i][j][k].synapse["input"] = FloatType(0)
					} else {
						p.axon[i][j][k].synapse["input"] = biasBool(true)
					}
				} else {
					if k < len(p.axon[i-1]) {
						p.axon[i][j][k].synapse["input"] = p.neuron[i-1][k]
					} else {
						p.axon[i][j][k].synapse["input"] = biasBool(true)
					}
				}
				p.axon[i][j][k].synapse["output"] = p.neuron[i][j]
			}
		}
	}
}*/

// initSynapseInput
/*func (p *perceptron) initSynapseInput(input []float64) {
	for j, w := range p.axon[0] {
		for k := range w {
			if k < p.lenInput {
				p.axon[0][j][k].synapse["input"] = FloatType(input[k])
			}
		}
	}
}*/

// calcNeuron
func (p *perceptron) calcNeuron(input []float64) {
	//p.initSynapseInput(input)
	//wait := make(chan bool)
	//defer close(wait)

	var length int
	//var neuron FloatType
	for i, v := range p.neuron {
		if i == 0 {
			length = p.lenInput
		} else {
			length = len(p.neuron[i-1])
		}
		for j, n := range v {
			//var sum FloatType = 0
			n = 0
			for k, weight := range p.Weights[i][j] {
				/*if i == 0 {
					if k < length {
						n += FloatType(input[k]) * weight
						neuron = FloatType(input[k])
					} else {
						n += weight
					}
				} else {
					if k < length {
						n += p.neuron[i-1][k] * weight
						neuron = p.neuron[i-1][k]
					} else {
						n += weight
					}
				}*/
				if k < length {
					if i == 0 {
						n += FloatType(input[k]) * weight
					} else {
						n += p.neuron[i-1][k] * weight
					}
				} else {
					n += weight
				}
			}
			fmt.Println(n, p.neuron[i][j])
			n = FloatType(calcActivation(float64(n), p.Activation))

			/*go func(n *neuron) {
				n.value = 0
				for _, a := range n.axon {
					n.value += a.getSynapseInput() * a.weight
				}
				n.value = FloatType(calcActivation(float64(n.value), p.Activation))
				wait <- true
			}(w)*/
		}
		/*for range v {
			<-wait
		}*/
	}
}

// calcLoss calculating the error of the output neuron
func (p *perceptron) calcLoss(target []float64) (loss float64) {
	for i, n := range p.neuron[p.lastIndexLayer] {
		//n.miss = FloatType(target[i]) - n.value
		p.miss[p.lastIndexLayer][i] = FloatType(target[i]) - n
		switch p.Loss {
		default:
			fallthrough
		case ModeMSE, ModeRMSE:
			loss += math.Pow(float64(p.miss[p.lastIndexLayer][i]), 2)
		case ModeARCTAN:
			loss += math.Pow(math.Atan(float64(p.miss[p.lastIndexLayer][i])), 2)
		}
		p.miss[p.lastIndexLayer][i] *= FloatType(calcDerivative(float64(p.miss[p.lastIndexLayer][i]), p.Activation))
	}
	loss /= float64(p.lenOutput)
	if p.Loss == ModeRMSE {
		loss = math.Sqrt(loss)
	}
	return
}

// calcMiss calculating the error of neurons in hidden layers
func (p *perceptron) calcMiss() {
	wait := make(chan bool)
	defer close(wait)
	for i := p.lastIndexLayer - 1; i >= 0; i-- {
		for j, v := range p.neuron[i] {
			go func(j int, n *neuron) {
				n.miss = 0
				for _, w := range p.neuron[i+1] {
					n.miss += w.miss * w.axon[j].weight
				}
				n.miss *= FloatType(calcDerivative(float64(n.value), p.Activation))
				wait <- true
			}(j, v)
		}
		for range p.neuron[i] {
			<-wait
		}
	}
}

// calcAxon update weights
func (p *perceptron) calcAxon() {
	p.calcMiss()
	wait := make(chan bool)
	defer close(wait)
	for _, u := range p.axon {
		for _, v := range u {
			for _, w := range v {
				go func(a *axon) {
					if n, ok := a.synapse["output"].(*neuron); ok {
						a.weight += a.getSynapseInput() * n.miss * p.Rate
					}
					wait <- true
				}(w)
			}
			for range v {
				<-wait
			}
		}
	}
}

// Train training neural network
func (p *perceptron) Train(input []float64, target ...[]float64) (loss float64, count int) {
	if !p.isInit {
		if p.isInit = p.init(len(input), getLengthData(target...)...); !p.isInit {
			pkg.LogError(fmt.Errorf("%w for train", pkg.ErrInit))
			return -1, 0
		}
	}
	_ = copy(p.input, input)
	_ = copy(p.target, target[0])
	if len(target) > 0 {
		for count < MaxIteration {
			p.calcNeuron(input)
			if loss = p.calcLoss(target[0]); loss <= p.Limit || loss <= MinLossLimit {
				break
			}
			p.calcMiss()
			p.calcAxon()
			count++
		}
		if count > 0 {
			p.isTrain = true
		}
	} else {
		pkg.LogError(pkg.ErrNoTarget)
		return -1, 0
	}
	return
}

// Query querying neural network
func (p *perceptron) Query(input []float64) (output []float64) {
	if !p.isTrain {
		pkg.LogError(fmt.Errorf("query: %w", pkg.ErrNotTrained))
		if !p.isInit {
			pkg.LogError(fmt.Errorf("%w for query", pkg.ErrInit))
			return nil
		}
	}
	p.calcNeuron( /*input*/ )
	output = make([]float64, p.lenOutput)
	for i, n := range p.neuron[p.lastIndexLayer] {
		output[i] = float64(n.value)
	}
	return
}

// Verify verifying neural network
func (p *perceptron) Verify(input []float64, target ...[]float64) (loss float64) {
	if !p.isTrain {
		pkg.LogError(fmt.Errorf("verify: %w", pkg.ErrNotTrained))
		if !p.isInit {
			if p.isInit = p.init(len(input), getLengthData(target...)...); !p.isInit {
				pkg.LogError(fmt.Errorf("%w for verify", pkg.ErrInit))
				return -1
			}
		}
	}
	if len(target) > 0 {
		p.calcNeuron(input)
		loss = p.calcLoss(target[0])
	} else {
		pkg.LogError(pkg.ErrNoTarget)
		return -1
	}
	return
}

// initWeight
/*func (p *perceptron) initWeight() {
	p.Weights = make(Float3Type, len(p.axon))
	for i, v := range p.axon {
		p.Weights[i] = make(Float2Type, len(p.axon[i]))
		for j := range v {
			p.Weights[i][j] = make(Float1Type, len(p.axon[i][j]))
		}
	}
	p.weight = &weight{
		isInitWeight: true,
		buffer:       &p.Weights,
	}
}*/

// getWeight
/*func (p *perceptron) getWeight() *Float3Type {
	p.copyWeight()
	return &p.Weights
}*/

// setWeight
/*func (p *perceptron) setWeight(weight *Float3Type) {
	for i, u := range *weight {
		for j, v := range u {
			for k, w := range v {
				p.axon[i][j][k].weight = w
			}
		}
	}
}*/

// copyWeight copies weights to the buffer
/*func (p *perceptron) copyWeight() {
	if p.weight == nil {
		p.initWeight()
	}
	for i, u := range p.axon {
		for j, v := range u {
			for k, w := range v {
				p.Weights[i][j][k] = w.weight
			}
		}
	}
}*/

// pasteWeight inserts weights from the buffer
/*func (p *perceptron) pasteWeight() (err error) {
	if p.Weights != nil {
		for i, u := range p.Weights {
			for j, v := range u {
				for k, w := range v {
					p.axon[i][j][k].weight = w
				}
			}
		}
		p.deleteWeight()
	} else {
		err = fmt.Errorf("paste weight error: missing weights")
	}
	return
}*/

// deleteWeight
/*func (p *perceptron) deleteWeight() {
	p.Weights = nil
	p.weight = nil
}*/

// readJSON
/*func (p *perceptron) readJSON(value interface{}) {
	if b, err := json.Marshal(&value); err != nil {
		pkg.LogError(fmt.Errorf("read marshal %w", err))
	} else if err = json.Unmarshal(b, &p); err != nil {
		pkg.LogError(fmt.Errorf("read unmarshal %w", err))
	}
	p.reInit()
	if err := p.pasteWeight(); err != nil {
		pkg.LogError(fmt.Errorf("read json: %w", err))
	}
}*/

// writeReport report of neural network training results in io.Writer
func (p *perceptron) writeReport(rep *report) {
	s := "----------------------------------------------\n"
	n := "\n"
	m := "\n\n"
	b := bytes.NewBufferString("Report of Perceptron Neural Network\n\n")

	printFormat := func(format string, a ...interface{}) {
		if _, err := fmt.Fprintf(b, format, a...); err != nil {
			pkg.LogError(fmt.Errorf("write report error: %w", err))
		}
	}

	// Input layer
	if in, ok := rep.args[0].([]float64); ok {
		printFormat("%s0 Input layer size: %d\n%sNeurons:\t", s, p.lenInput, s)
		for _, v := range in {
			printFormat("  %v", v)
		}
		printFormat("%s", m)
	}

	// Layers: neuron, miss
	var t string
	for i, v := range p.neuron {
		switch i {
		case p.lastIndexLayer:
			t = "Output layer"
		default:
			t = "Hidden layer"
		}
		printFormat("%s%d %s size: %d\n%sNeurons:\t", s, i+1, t, len(p.neuron[i]), s)
		for _, w := range v {
			printFormat("  %11.8f", w.value)
		}
		printFormat("\nMiss:\t\t")
		for _, w := range v {
			printFormat("  %11.8f", w.miss)
		}
		printFormat("%s", m)
	}

	// Axons: weight
	printFormat("%sAxons (weights)\n%s", s, s)
	for _, u := range p.axon {
		for i, v := range u {
			printFormat("%d", i+1)
			for _, w := range v {
				printFormat("\t%11.8f", w.weight)
			}
			printFormat("%s", n)
		}
		printFormat("%s", n)
	}

	// Resume
	if loss, ok := rep.args[1].(float64); ok {
		printFormat("%sTotal loss (error):\t\t%v\n", s, loss)
	}
	if count, ok := rep.args[2].(int); ok {
		printFormat("Number of iteration:\t%v\n", count)
	}

	if _, err := b.WriteTo(rep.file); err != nil {
		pkg.LogError(fmt.Errorf("write report error: %w", err))
	} else if err = rep.file.Close(); err != nil {
		pkg.LogError(fmt.Errorf("write report close error: %w", err))
	}
}
