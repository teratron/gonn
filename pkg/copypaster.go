package pkg

type CopyPaster interface {
	Copy(Getter)
	Paste(Getter) error
}