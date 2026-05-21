package loss

import "github.com/teratron/gonn/pkg/utils"

// For single value, return 0 as CCE requires vectors
func cceLossSingle[T utils.Float](predicted, target T) T {
	return 0
}
