package nn

import (
	"log"
	"runtime"
)

type modeLogType uint8

type Logger interface {
}

func Logging(mode ...modeLogType) GetterSetter {
	if len(mode) == 0 {
		return modeLossType(0)
	} else {
		return mode[0]
	}
}

// Setter
func (l modeLogType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if n, ok := args[0].(*nn); ok {
			n.logging = l
		}
	}
}

// Getter
func (l modeLogType) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return l
	} else {
		if n, ok := args[0].(*nn); ok {
			return n.logging
		}
	}
	return nil
}

//
func Log(reason string, info bool) {
	if !info {
		log.Println(reason)
	} else if _, file, line, ok := runtime.Caller(1); ok {
		log.Printf("%v, at: %s:%d\n", reason, file, line)
	}
}

// LogError reports an error to the command line with the specified err cause,
// if not nil.
// The function also reports basic information about the code location.
func LogError(reason string, err error) {
	log.Println("Error: ", reason)
	if err != nil {
		log.Println("\tCause:", err)
	}
	if _, file, line, ok := runtime.Caller(1); ok {
		log.Printf("\tAt: %s:%d", file, line)
	}
}