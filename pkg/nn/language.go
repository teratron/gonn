package nn

type langType string

func Language(lang ...langType) GetterSetter {
	if len(lang) == 0 {
		return langType("")
	} else {
		return lang[0]
	}
}

// Setter
func (l langType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if n, ok := args[0].(*nn); ok {
			n.language = l
		}
	}
}

// Getter
func (l langType) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return l
	} else {
		if n, ok := args[0].(*nn); ok {
			return n.language
		}
	}
	return nil
}