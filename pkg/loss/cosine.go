package loss

import "math"

// Cosine similarity/distance calculation for vectors
func cosineLossVector[T float32 | float64](predicted, target []T) T {
	if len(predicted) != len(target) || len(predicted) == 0 {
		return 0
	}

	var dotProduct, normPred, normTarget T
	for i := 0; i < len(predicted); i++ {
		dotProduct += predicted[i] * target[i]
		normPred += predicted[i] * predicted[i]
		normTarget += target[i] * target[i]
	}

	normPred = T(math.Sqrt(float64(normPred)))
	normTarget = T(math.Sqrt(float64(normTarget)))

	if normPred == 0 || normTarget == 0 {
		// If one of the vectors is zero, return maximum distance (1)
		return 1
	}

	cosineSimilarity := dotProduct / (normPred * normTarget)
	// Cosine distance = 1 - cosine similarity
	return 1 - cosineSimilarity
}
