# Example: Iris Multi-class Classification

Classification of the classic Fisher Iris dataset into three species using a multi-layer perceptron. This example demonstrates multi-class classification with the `SoftMax` activation and `MSE` loss.

## Overview

The network takes 4 physical measurements of an iris flower and predicts which of the 3 species it belongs to: *Iris setosa*, *Iris versicolor*, or *Iris virginica*.

> [!NOTE]
> This example uses `go:embed` to bundle the `iris.csv` dataset directly into the binary, making it fully portable.

## Network Topology

The architecture follows a standard multi-hidden layer setup:

- **Input**: 4 features (Sepal Length, Sepal Width, Petal Length, Petal Width)
- **Hidden Layer 1**: 16 neurons with `ReLU` activation
- **Hidden Layer 2**: 8 neurons with `ReLU` activation
- **Output**: 3 neurons with `SoftMax` activation (one-hot encoded species)

## Mathematical Foundation

The network is trained to minimize the Mean Squared Error (MSE) between the predicted probabilities and the one-hot encoded ground truth:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:

- $y_i$ is the target one-hot vector.
- $\hat{y}_i$ is the predicted probability vector from `SoftMax`.

## Features

- **Data Splitting**: Uses an 80/20 train/test split.
- **Normalization**: Features are used in their raw scale (demonstrating `Xavier` initialization stability).
- **Callbacks**: Uses `WithEpochCallback` to monitor training progress.
- **Accuracy**: Reports final classification accuracy for both training and hold-out sets.

## Running the example

```bash
go run ./examples/iris
```

## Running the tests

```bash
go test ./examples/iris
```
