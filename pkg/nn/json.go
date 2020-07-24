package nn

type json struct {
	fileName string
}

func JSON(filename ...string) *json {
	return &json{}
}