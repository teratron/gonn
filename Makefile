
hello:
	echo "Hello"

build:
	go build -o cmd/nn/main main.go

run:
	go run main.go

compile:
	echo "Compiling for every OS and Platform"
    # 32-Bit Systems
    # FreeBDS
    GOOS=freebsd GOARCH=386 go build -o bin/main-freebsd-386 main.go
    # MacOS
    GOOS=darwin GOARCH=386 go build -o bin/main-darwin-386 main.go
    # Linux
    GOOS=linux GOARCH=386 go build -o bin/main-linux-386 main.go
    # Windows
    GOOS=windows GOARCH=386 go build -o bin/main-windows-386 main.go
        # 64-Bit
    # FreeBDS
    GOOS=freebsd GOARCH=amd64 go build -o bin/main-freebsd-amd64 main.go
    # MacOS
    GOOS=darwin GOARCH=amd64 go build -o bin/main-darwin-amd64 main.go
    # Linux
    GOOS=linux GOARCH=amd64 go build -o bin/main-linux-amd64 main.go
    # Windows
    GOOS=windows GOARCH=amd64 go build -o bin/main-windows-amd64 main.go

all:
	hello build
