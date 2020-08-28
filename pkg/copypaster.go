//
package pkg

type CopyPaster interface {
	Copier
	Paster
}

type Copier interface {
	Copy(Copier)
}

type Paster interface {
	Paste(Paster) error
}
