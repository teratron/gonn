//
package pkg

type GetterSetter interface {
	Getter
	Setter
}

type Getter interface {
	Get(...Getter) GetterSetter
}

type Setter interface {
	Set(...Setter)
}