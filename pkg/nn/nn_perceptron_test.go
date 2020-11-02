package nn

import (
	"reflect"
	"testing"

	"github.com/zigenzoog/gonn/pkg"
)

func TestPerceptron(t *testing.T) {
	tests := []struct {
		name string
		want *perceptron
	}{
		// TODO: Add test cases.
		{
			name: "a",
			want: &perceptron{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Perceptron(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Perceptron() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_ActivationMode(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   uint8
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.ActivationMode(); got != tt.want {
				t.Errorf("ActivationMode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_Bias(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.Bias(); got != tt.want {
				t.Errorf("Bias() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_Copy(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		copier pkg.Copier
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_Get(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		args []pkg.Getter
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   pkg.GetSetter
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.Get(tt.args.args...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Get() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_HiddenLayer(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   []uint
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.HiddenLayer(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("HiddenLayer() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_LossLevel(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.LossLevel(); got != tt.want {
				t.Errorf("LossLevel() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_LossMode(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   uint8
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.LossMode(); got != tt.want {
				t.Errorf("LossMode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_Paste(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		paster pkg.Paster
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_Query(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		input []float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantOutput []float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if gotOutput := p.Query(tt.args.input); !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("Query() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}

func Test_perceptron_Rate(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   float32
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.Rate(); got != tt.want {
				t.Errorf("Rate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_Read(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		reader pkg.Reader
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_Set(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		args []pkg.Setter
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_Train(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		input  []float64
		target [][]float64
	}
	tests := []struct {
		name      string
		fields    fields
		args      args
		wantLoss  float64
		wantCount int
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			gotLoss, gotCount := p.Train(tt.args.input, tt.args.target...)
			if gotLoss != tt.wantLoss {
				t.Errorf("Train() gotLoss = %v, want %v", gotLoss, tt.wantLoss)
			}
			if gotCount != tt.wantCount {
				t.Errorf("Train() gotCount = %v, want %v", gotCount, tt.wantCount)
			}
		})
	}
}

func Test_perceptron_Verify(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		input  []float64
		target [][]float64
	}
	tests := []struct {
		name     string
		fields   fields
		args     args
		wantLoss float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if gotLoss := p.Verify(tt.args.input, tt.args.target...); gotLoss != tt.wantLoss {
				t.Errorf("Verify() = %v, want %v", gotLoss, tt.wantLoss)
			}
		})
	}
}

func Test_perceptron_Weight(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   Floater
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.Weight(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Weight() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_Write(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		writer []pkg.Writer
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_architecture(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   Architecture
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.architecture(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("architecture() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_calcAxon(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		input []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_calcLoss(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		target []float64
	}
	tests := []struct {
		name     string
		fields   fields
		args     args
		wantLoss float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if gotLoss := p.calcLoss(tt.args.target); gotLoss != tt.wantLoss {
				t.Errorf("calcLoss() = %v, want %v", gotLoss, tt.wantLoss)
			}
		})
	}
}

func Test_perceptron_calcMiss(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		input []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_calcNeuron(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		input []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_copyWeight(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_deleteWeight(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_getWeight(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
		want   *float3Type
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if got := p.getWeight(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getWeight() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_perceptron_initAxon(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_initNeuron(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_initSynapseInput(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		input []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_initWeight(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_pasteWeight(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name    string
		fields  fields
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
			if err := p.pasteWeight(); (err != nil) != tt.wantErr {
				t.Errorf("pasteWeight() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_perceptron_reInit(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_readCSV(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_readJSON(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		value interface{}
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_setArchitecture(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		network Architecture
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_setWeight(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		weight *float3Type
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_writeCSV(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		filename csvString
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}

func Test_perceptron_writeReport(t *testing.T) {
	type fields struct {
		Architecture Architecture
		Parameter    Parameter
		Constructor  Constructor
		Conf         struct {
			// Array of the number of neurons in each hidden layer
			HiddenLayer HiddenArrUint `json:"hiddenLayer" xml:"hiddenLayer>layer"`

			// The neuron bias, false or true
			Bias biasBool `json:"bias" xml:"bias"`

			// Activation function mode
			ActivationMode uint8 `json:"activationMode" xml:"activationMode"`

			// The mode of calculation of the total error
			LossMode uint8 `json:"lossMode" xml:"lossMode"`

			// Minimum (sufficient) level of the average of the error during training
			LossLevel float64 `json:"lossLevel" xml:"lossLevel"`

			// Learning coefficient, from 0 to 1
			Rate floatType `json:"rate" xml:"rate"`

			// Buffer of weight values
			Weight float3Type `json:"weight" xml:"weight>weight"`
		}
		neuron         [][]*neuron
		axon           [][][]*axon
		weight         *weight
		lastIndexLayer int
		lenInput       int
		lenOutput      int
	}
	type args struct {
		rep *report
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &perceptron{
				Architecture:   tt.fields.Architecture,
				Parameter:      tt.fields.Parameter,
				Constructor:    tt.fields.Constructor,
				Conf:           tt.fields.Conf,
				neuron:         tt.fields.neuron,
				axon:           tt.fields.axon,
				weight:         tt.fields.weight,
				lastIndexLayer: tt.fields.lastIndexLayer,
				lenInput:       tt.fields.lenInput,
				lenOutput:      tt.fields.lenOutput,
			}
		})
	}
}