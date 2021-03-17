
run-example:
	go run ./examples/main.go

run-example-hopfield:
	go run ./examples/hopfield/main.go

run-example-tmp:
	go run ./examples/tmp/demo.go

mods:
	go get -u gopkg.in/yaml.v2
	go mod tidy
	go mod vendor