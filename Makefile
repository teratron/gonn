
run-example:
	go run -v ./examples/main.go

run-example-hopfield:
	go run -v ./examples/hopfield/main.go

run-example-query:
	go run -v ./examples/query/main.go

run-example-json:
	go run -v ./examples/json/main.go

run-example-yaml:
	go build -v -mod vendor ./examples/yaml/main.go

deps:
	go get
	go mod verify
	go mod tidy -v
	go mod vendor -v

clean:
	go clean -modcache