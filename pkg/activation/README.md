# Activation Functions and their Derivatives

| Identifier    | Description                                                 | Parameters         |
|---------------|-------------------------------------------------------------|--------------------|
| ELISH         | Exponential Linear Unit + Sigmoid                           |                    |
| ELU           | Exponential Linear Unit                                     | `alpha` (default 1.0) |
| LINEAR        | Linear/identity                                             | `slope` (default 1.0), `offset` (default 0.0) |
| LEAKY_RELU    | Leaky Rectified Linear Unit                                 | `leak` (default 0.01) |
| RELU          | Rectified Linear Unit                                       |                    |
| SELU          | Scaled Exponential Linear Unit                              | `scale` (default 1.0507), `alpha` (default 1.6733) |
| SIGMOID       | Logistic, a.k.a. sigmoid or soft step                       | `slope` (default 1.0) |
| SOFTMAX       | Softmax (Note: Current implementation is a placeholder for single values, requires vector for full functionality) |                    |
| SWISH         | Swish-function                                              | `beta` (default 1.0) |
| TANH          | Hyperbolic Tangent                                          |                    |
