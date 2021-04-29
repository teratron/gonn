
test:
	go test -v -cover ./pkg/... ./internal/...

build-example-perceptron:
	go build -v -o ./examples/perceptron ./examples/perceptron/main.go

build-example-hopfield:
	go build -v -o ./examples/hopfield ./examples/hopfield/main.go

build-example-linear:
	go build -v -o ./examples/linear ./examples/linear/main.go

build-example-query:
	go build -v -o ./examples/query ./examples/query/main.go

build-example-json:
	go build -v -o ./examples/json ./examples/json/main.go

build-example-yaml:
	go build -v -mod vendor -o ./examples/yaml ./examples/yaml/main.go

setup:
	go mod init
	go mod tidy -v
	go mod vendor -v

deps:
	go mod verify
	go mod tidy -v
	go mod vendor -v

clean:
	go clean -modcache

#VERSION := $(shell cat ./VERSION)

release:
	git tag -a $(VERSION) -m "Release" || true
	git push origin $(VERSION)
	goreleaser --rm-dist