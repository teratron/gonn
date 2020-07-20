package nn

type langType string

/*func Language(lang ...langType) GetterSetter {
	if len(lang) > 0 {
		return lang[0]
	} else {
		return langType("")
	}
}

// Setter
func (l langType) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(*nn); ok {
			n.language = l
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (l langType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if n, ok := args[0].(*nn); ok {
			return n.language
		}
	} else {
		return l
	}
	return nil
}*/