package pkg

import "testing"

func TestFloat1Type_length(t *testing.T) {
	tests := []struct {
		name  string
		index []uint
		gave  Float1Type
		want  int
	}{
		{
			name:  "#1",
			index: []uint{},
			gave:  Float1Type{1, 2},
			want:  2,
		},
		{
			name:  "#2",
			index: []uint{5},
			gave:  Float1Type{1, 2, 3},
			want:  3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.Length(tt.index...); got != tt.want {
				t.Errorf("length() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestFloat2Type_length(t *testing.T) {
	tests := []struct {
		name  string
		index []uint
		gave  Float2Type
		want  int
	}{
		{
			name:  "#1",
			index: []uint{1},
			gave: Float2Type{
				{1},
				{1, 2},
			},
			want: 2,
		},
		{
			name:  "#2_no_args",
			index: []uint{},
			gave: Float2Type{
				{1},
				{1, 2},
			},
			want: 2,
		},
		{
			name:  "#3_overflow",
			index: []uint{3, 1},
			gave: Float2Type{
				{1},
				{1, 2},
			},
			want: 0,
		},
		{
			name:  "#4_empty_array",
			index: []uint{2},
			gave:  Float2Type{},
			want:  0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.Length(tt.index...); got != tt.want {
				t.Errorf("length() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestFloat3Type_length(t *testing.T) {
	tests := []struct {
		name  string
		index []uint
		gave  Float3Type
		want  int
	}{
		{
			name:  "#1",
			index: []uint{1, 2},
			gave: Float3Type{
				{
					{1, 2},
					{1, 2, 3},
				},
				{
					{1},
					{1, 2},
					{1, 2, 3},
				},
			},
			want: 3,
		},
		{
			name:  "#2",
			index: []uint{0},
			gave: Float3Type{
				{
					{1, 2},
					{1, 2, 3},
				},
				{
					{1},
					{1, 2},
					{1, 2, 3},
				},
			},
			want: 2,
		},
		{
			name:  "#3_no_args",
			index: []uint{},
			gave: Float3Type{
				{
					{1, 2},
					{1, 2, 3},
				},
				{
					{1},
					{1, 2},
					{1, 2, 3},
				},
			},
			want: 2,
		},
		{
			name:  "#4_overflow",
			index: []uint{0, 3, 2},
			gave: Float3Type{
				{
					{1, 2},
					{1, 2, 3},
				},
				{
					{1},
					{1, 2},
					{1, 2, 3},
				},
			},
			want: 0,
		},
		{
			name:  "#5_empty_array",
			index: []uint{0, 3, 2},
			gave:  Float3Type{},
			want:  0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.gave.Length(tt.index...); got != tt.want {
				t.Errorf("length() = %d, want %d", got, tt.want)
			}
		})
	}
}
