package nn

type langType string

func Language(lang ...langType) Setter {
	if len(lang) == 0 {
		return langType("")
	} else {
		return lang[0]
	}
}

// Setter
func (l langType) Set(set ...Setter) {
	if n, ok := set[0].(*nn); ok {
		n.language = l
	}
}

// Getter
func (l langType) Get(set ...Setter) Getter {
	if n, ok := set[0].(*nn); ok {
		return n.language
	}
	return nil
}