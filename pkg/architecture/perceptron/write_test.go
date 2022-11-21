package perceptron

import (
	"fmt"
	"os"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

var tmpJSON = "tmp.json"

func TestNN_WriteConfig(t *testing.T) {
	testErr := fmt.Errorf("error")
	tests := []struct {
		name    string
		args    []string
		gave    *NN
		wantErr error
	}{
		{
			name:    "#1_" + tmpJSON,
			args:    []string{tmpJSON},
			gave:    &NN{},
			wantErr: nil,
		},
		{
			name:    "#2_no_args_" + tmpJSON,
			args:    []string{},
			gave:    &NN{config: &utils.FileJSON{Name: tmpJSON}},
			wantErr: nil,
		},
		{
			name:    "#3_no_args",
			args:    []string{},
			gave:    &NN{},
			wantErr: testErr,
		},
		{
			name:    "#4_error_write",
			args:    []string{"."},
			gave:    &NN{},
			wantErr: testErr,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErr := tt.gave.WriteConfig(tt.args...)
			if gotErr == nil {
				defer func() {
					if len(tt.args) == 0 {
						tt.args = []string{tt.gave.config.GetName()}
					}
					if err := os.Remove(tt.args[0]); err != nil {
						t.Error(err)
					}
				}()
			}
			if (gotErr != nil && tt.wantErr == nil) || (gotErr == nil && tt.wantErr != nil) {
				t.Errorf("WriteConfig()\ngot error:\t%v\nwant error:\t%v", gotErr, tt.wantErr)
			}
		})
	}
}

func TestNN_WriteWeight(t *testing.T) {
	testErr := fmt.Errorf("error")
	tests := []struct {
		name    string
		args    string
		gave    *NN
		wantErr error
	}{
		{
			name:    "#1_" + tmpJSON,
			args:    tmpJSON,
			gave:    &NN{},
			wantErr: nil,
		},
		{
			name:    "#2_no_args_error_write",
			args:    "",
			gave:    &NN{},
			wantErr: testErr,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotErr := tt.gave.WriteWeight(tt.args)
			if gotErr == nil {
				defer func() {
					if err := os.Remove(tt.args); err != nil {
						t.Error(err)
					}
				}()
			}
			if (gotErr != nil && tt.wantErr == nil) || (gotErr == nil && tt.wantErr != nil) {
				t.Errorf("WriteWeight()\ngot error:\t%v\nwant error:\t%v", gotErr, tt.wantErr)
			}
		})
	}
}
