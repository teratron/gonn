# Perceptron Neural Network

### Name

Neural network architecture name (required field for a config).

### Bias

The neuron bias, false or true (required field for a config).

### Hidden

Array of the number of neurons in each hidden layer.

### Activation

Activation function mode (required field for a config).

| Code | Activation | Description                               |
| ---- | ---------- | ----------------------------------------- |
| 0    | LINEAR     | Linear/identity.                          |
| 1    | RELU       | ReLu (rectified linear unit).             |
| 2    | LEAKYRELU  | Leaky ReLu (leaky rectified linear unit). |
| 3    | SIGMOID    | Logistic, a.k.a. sigmoid or soft step.    |
| 4    | TANH       | TanH (hyperbolic tangent).                |

### Loss

The mode of calculation of the total error.

| Code | Loss   | Description              |
| ---- | ------ | ------------------------ |
| 0    | MSE    | Mean Squared Error.      |
| 1    | RMSE   | Root Mean Squared Error. |
| 2    | ARCTAN | Arctan.                  |
| 3    | AVG    | Average.                 |

### Rate

Learning coefficient (greater than 0 and less than or equal to 1).
