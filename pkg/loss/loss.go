package loss

import "math"

// LossType represents different loss functions.
type Type uint8

// Loss function mode.
const (
	MSE       Type = iota // MSE - Mean Squared Error
	MAE                   // MAE - Mean Absolute Error (equivalent to Avg)
	CCE                   // CCE - Categorical Cross-Entropy
	BCE                   // BCE - Binary Cross-Entropy
	MAPE                  // MAPE - Mean Absolute Percentage Error
	MSLE                  // MSLE - Mean Squared Logarithmic Error
	KLD                   // KLD - Kullback-Leibler Divergence
	COSINE                // COSINE - Cosine Similarity/Distance
	POISSON               // POISSON - Poisson Loss Function
	HINGE                 // HINGE - Hinge Loss
	SQ_HINGE              // SQ_HINGE - Squared Hinge Loss
	CAT_HINGE             // CAT_HINGE - Categorical Hinge Loss
	LOG_COSH              // LOG_COSH - Log-Cosh Loss
	HUBER                 // HUBER - Huber Loss
	AVG                   // AVG - Average Error (Mean Absolute Error)
	RMSE                  // RMSE - Root Mean Squared Error
	ARCTAN                // ARCTAN - Arctan Error
)

// Loss function for single values.
func Loss[T float32 | float64](predicted, target T, mode Type, params ...float64) T {
	switch mode {
	case MSE:
		return mseLoss(predicted, target)
	case MAE:
		return maeLoss(predicted, target)
	case AVG:
		return avgLoss(predicted, target)
	case RMSE:
		return rmseLoss(predicted, target)
	case ARCTAN:
		return arctanLoss(predicted, target)
	case BCE:
		return bceLoss(predicted, target)
	case MAPE:
		return mapeLoss(predicted, target)
	case MSLE:
		return msleLoss(predicted, target)
	case KLD:
		return kldLoss(predicted, target)
	case CCE:
		return cceLossSingle(predicted, target) // CCE requires vector inputs, so return 0 for single values
	case POISSON:
		return poissonLoss(predicted, target)
	case HINGE:
		return hingeLoss(predicted, target)
	case SQ_HINGE:
		return sqHingeLoss(predicted, target)
	case CAT_HINGE:
		return catHingeLoss(predicted, target)
	case LOG_COSH:
		return logCoshLoss(predicted, target)
	case HUBER:
		return huberLoss(predicted, target)
	default:
		// For unimplemented functions, return appropriate defaults
		// For now, return MSE as default for unimplemented functions
		return mseLoss(predicted, target)
	}
}

// LossVector function for vector inputs (slices).
func LossVector[T float32 | float64](predicted, target []T, mode Type, params ...float64) T {
	if len(predicted) != len(target) {
		// Return zero if slices have different lengths
		return T(0)
	}

	switch mode {
	case CCE:
		// CCE has special implementation for vectors
		return cceLoss(predicted, target)
	case COSINE:
		// Cosine similarity/distance needs special handling
		return cosineLossVector(predicted, target)
	default:
		// For other functions, calculate as before
		var total T
		n := len(predicted)
		for i := 0; i < n; i++ {
			total += Loss(predicted[i], target[i], mode, params...)
		}

		switch mode {
		case MSE, MSLE, LOG_COSH:
			// For squared error functions, return the mean
			return total / T(n)
		case RMSE:
			// For RMSE, return the square root of the mean squared error
			return T(math.Sqrt(float64(total / T(n))))
		case MAE, AVG, MAPE, ARCTAN:
			// For absolute error functions, return the mean
			return total / T(n)
		case BCE, KLD, POISSON, HINGE, SQ_HINGE, CAT_HINGE, HUBER:
			// For other loss functions, return the mean
			return total / T(n)
		default:
			return total / T(n)
		}
	}
}
