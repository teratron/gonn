// GetterSetter
package pkg

type GetSetter interface {
	Getter
	Setter
}

type Getter interface {
	Get(...Getter) GetSetter
}

type Setter interface {
	Set(...Setter)
}