
run-example:
	go run ./examples/main.go

run-example-hopfield:
	go run ./examples/hopfield/main.go

run-example-json:
	go run ./examples/json/main.go

run-example-yaml:
	go run ./examples/yaml/main.go

deps:
	go get -u gopkg.in/yaml.v2
	go mod tidy
	go mod vendor