# Perceptron Neural Network

### Name

Neural network architecture name (required field for a config).

### Bias

The neuron bias, false or true (required field for a config).

### Hidden

Array of the number of neurons in each hidden layer.

### Activation

Activation function mode (required field for a config).

| Code | Activation    | Description                               |
| ---- | ------------- | ----------------------------------------- |
| 0    | ModeLINEAR    | Linear/identity.                          |
| 1    | ModeRELU      | ReLu (rectified linear unit).             |
| 2    | ModeLEAKYRELU | Leaky ReLu (leaky rectified linear unit). |
| 3    | ModeSIGMOID   | Logistic, a.k.a. sigmoid or soft step.    |
| 4    | ModeTANH      | TanH (hyperbolic tangent).                |

### Loss

The mode of calculation of the total error.

| Code | Loss       | Description              |
| ---- | ---------- | ------------------------ |
| 0    | ModeMSE    | Mean Squared Error.      |
| 1    | ModeRMSE   | Root Mean Squared Error. |
| 2    | ModeARCTAN | Arctan.                  |
| 3    | ModeAVG    | Average.                 |



### Limit

Minimum (sufficient) limit of the average of the error during training.

### Rate

Learning coefficient (greater than 0 and less than or equal to 1).

	// DefaultRate default learning rate.
	DefaultRate = 0.3