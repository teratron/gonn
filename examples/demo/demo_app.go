package main

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg/app"
)

func main() {
	// Application
	a := app.App()
	fmt.Println("nn.App():", a)

	// Common
	/*a.Set(nn.Language("ru"),
	  nn.Logging(1))*/

	// Language
	//n.Set(nn.Language("ru"))
	//fmt.Println("n.Get(nn.Language()):", n.Get(nn.Language()))

	// Logging
	//n.Set(nn.Logging(0)) //set
	//fmt.Println("n.Get(nn.Logging()):", n.Get(nn.Logging()))
}
