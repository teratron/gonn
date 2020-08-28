package pkg

type GetSetter interface {
	Getter
	Setter
}

// Get
type Getter interface {
	Get(...Getter) GetSetter
}

// Set
type Setter interface {
	Set(...Setter)
}
