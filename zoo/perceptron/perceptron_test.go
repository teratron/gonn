package perceptron

import (
	"reflect"
	"testing"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/params"
)

func init() {
	GetMaxIteration = func() int { return 1 }
	params.GetRandFloat = func() float64 { return .5 }
}

func Test_getMaxIteration(t *testing.T) {
	t.Run("MaxIteration", func(t *testing.T) {
		if got := getMaxIteration(); got != MaxIteration {
			t.Errorf("getMaxIteration() = %v, want %v", got, MaxIteration)
		}
	})
}

func TestNew(t *testing.T) {
	want := &NN{
		Name:       Name,
		Activation: params.ModeSIGMOID,
		Loss:       params.ModeMSE,
		Limit:      .1,
		Rate:       params.DefaultRate,
	}
	t.Run(want.Name, func(t *testing.T) {
		if got := New(); !reflect.DeepEqual(got, want) {
			t.Errorf("New()\ngot:\t%v\nwant:\t%v", got, want)
		}
	})
}

/*func TestNN_Read(t *testing.T) {
	gave := &perceptron{}
	tests := []struct {
		name    string
		reader  Reader
		wantErr error
	}{
		{
			name:    "#1_type_missing_read",
			reader:  Reader(nil),
			wantErr: fmt.Errorf("error"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErr := gave.Read(tt.reader)
			if (gotErr != nil && tt.wantErr == nil) || (gotErr == nil && tt.wantErr != nil) {
				t.Errorf("Read()\ngot error:\t%v\nwant error:\t%v", gotErr, tt.wantErr)
			}
		})
	}
}

func TestNN_Write(t *testing.T) {
	existErr := fmt.Errorf("error")
	gave := &perceptron{}
	tests := []struct {
		name    string
		writer  []Writer
		wantErr error
	}{
		{
			name:    "#1",
			writer:  []Writer{JSON(defaultNameJSON)},
			wantErr: nil,
		},
		{
			name:    "#2_no_args",
			writer:  nil,
			wantErr: existErr,
		},
		{
			name:    "#3_type_missing_write",
			writer:  []Writer{nil},
			wantErr: existErr,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErr := gave.Write(tt.writer...)
			if len(tt.writer) > 0 {
				if w, ok := tt.writer[0].(jsonString); ok {
					defer func() {
						if err := os.Remove(string(w)); err != nil {
							t.Error(err)
						}
					}()
				}
			}
			if (gotErr != nil && tt.wantErr == nil) || (gotErr == nil && tt.wantErr != nil) {
				t.Errorf("Write()\ngot error:\t%v\nwant error:\t%v", gotErr, tt.wantErr)
			}
		})
	}
}*/

func TestNN_Train(t *testing.T) {
	type args struct {
		input  []float64
		target []float64
	}
	tests := []struct {
		name string
		args
		gave      *NN
		wantLoss  float64
		wantCount int
	}{
		{
			name: "#1",
			args: args{[]float64{.2, .3}, []float64{.2}},
			gave: &NN{
				Activation: params.ModeSIGMOID,
				Loss:       params.ModeMSE,
				Weights: gonn.Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuron{
					{
						&neuron{},
						&neuron{},
					},
					{
						&neuron{},
					},
				},
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
			},
			wantLoss:  .1236831826342541,
			wantCount: 1,
		},
		{
			name:      "#2_no_input",
			args:      args{input: []float64{}},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name:      "#3_no_target",
			args:      args{[]float64{.2}, []float64{}},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#4_error_len_input",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &NN{
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#5_error_len_target",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &NN{
				lenInput:  1,
				lenOutput: 2,
				isInit:    true,
			},
			wantLoss:  -1,
			wantCount: 0,
		},
		{
			name: "#6_not_init",
			args: args{[]float64{.2, .3}, []float64{.3}},
			gave: &NN{
				Bias:   true,
				Hidden: []int{2},
				Limit:  .95,
			},
			wantLoss:  .9025,
			wantCount: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotLoss, gotCount := tt.gave.Train(tt.input, tt.target)
			if gotLoss != tt.wantLoss {
				t.Errorf("Train() gotLoss = %f, wantLoss %f", gotLoss, tt.wantLoss)
			}
			if gotCount != tt.wantCount {
				t.Errorf("Train() gotCount = %d, wantCount %d", gotCount, tt.wantCount)
			}
		})
	}
}

func TestNN_Verify(t *testing.T) {
	type args struct {
		input  []float64
		target []float64
	}
	tests := []struct {
		name string
		args
		gave *NN
		want float64
	}{
		{
			name: "#1",
			args: args{[]float64{.2, .3}, []float64{.2}},
			gave: &NN{
				Activation: 255, // default ModeSIGMOID
				Loss:       255, // default ModeMSE
				Weights: gonn.Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuron{
					{
						&neuron{},
						&neuron{},
					},
					{
						&neuron{},
					},
				},
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
			},
			want: .1236831826342541,
		},
		{
			name: "#2_no_input",
			args: args{input: []float64{}},
			want: -1,
		},
		{
			name: "#3_no_target",
			args: args{[]float64{.2}, []float64{}},
			want: -1,
		},
		{
			name: "#4_error_len_input",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &NN{
				lenInput:  2,
				lenOutput: 1,
				isInit:    true,
			},
			want: -1,
		},
		{
			name: "#5_error_len_target",
			args: args{[]float64{.2}, []float64{.3}},
			gave: &NN{
				lenInput:  1,
				lenOutput: 2,
				isInit:    true,
			},
			want: -1,
		},
		{
			name: "#6_not_init",
			args: args{[]float64{.2, .3}, []float64{.3}},
			gave: &NN{
				Bias:   true,
				Hidden: []int{2},
			},
			want: .9025,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.Verify(tt.input, tt.target); got != tt.want {
				t.Errorf("Verify() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestNN_Query(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		gave  *NN
		want  []float64
	}{
		{
			name:  "#1",
			input: []float64{.2, .3},
			gave: &NN{
				Activation: params.ModeSIGMOID,
				Weights: gonn.Float3Type{
					{
						{.1, .1, .1},
						{.1, .1, .1},
					},
					{
						{.1, .1, .1},
					},
				},
				neuron: [][]*neuron{
					{
						&neuron{},
						&neuron{},
					},
					{
						&neuron{},
					},
				},
				lenInput:       2,
				lenOutput:      1,
				lastLayerIndex: 1,
				isInit:         true,
			},
			want: []float64{.5516861990955205},
		},
		{
			name:  "#2_no_input",
			input: []float64{},
			want:  nil,
		},
		{
			name:  "#3_not_init",
			input: []float64{.1},
			gave:  &NN{isInit: false},
			want:  nil,
		},
		{
			name:  "#4_error_len_input",
			input: []float64{.1},
			gave: &NN{
				lenInput: 2,
				isInit:   true,
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.Query(tt.input); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Query()\ngot:\t%v\nwant:\t%v", got, tt.want)
			}
		})
	}
}
