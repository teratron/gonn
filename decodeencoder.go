package gonn

// DecodeEncoder
type DecodeEncoder interface {
	Decoder
	Encoder
}

// Decoder
type Decoder interface {
	Decode(interface{}) error
}

// Encoder
type Encoder interface {
	Encode(interface{}) error
}

// Filer
type Filer interface {
	DecodeEncoder
	GetValue(key string) interface{}
}
