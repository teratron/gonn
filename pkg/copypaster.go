package pkg

// CopyPaster
type CopyPaster interface {
	Copier
	Paster
}

// Copier
type Copier interface {
	Copy(Copier)
}

// Paster
type Paster interface {
	Paste(Paster)
}