/* Dense Forward kernel — matmul + bias + sigmoid activation.
 * One work-item per output neuron i.
 *
 * weights[i * inSize + j] for j in [0, inSize)
 * bias[i]
 * input[j]  for j in [0, inSize)
 * output[i] = sigmoid(dot(weights[i], input) + bias[i])
 */

#define SIGMOID(x) (1.0f / (1.0f + exp(-(x))))

__kernel void dense_forward_f32(
    __global const float* weights,
    __global const float* bias,
    __global const float* input,
    __global       float* output,
    const int inSize,
    const int outSize
) {
    int i = get_global_id(0);
    if (i >= outSize) return;

    float acc = bias[i];
    for (int j = 0; j < inSize; j++) {
        acc += weights[i * inSize + j] * input[j];
    }
    output[i] = SIGMOID(acc);
}

__kernel void dense_forward_f64(
    __global const double* weights,
    __global const double* bias,
    __global const double* input,
    __global       double* output,
    const int inSize,
    const int outSize
) {
    int i = get_global_id(0);
    if (i >= outSize) return;

    double acc = bias[i];
    for (int j = 0; j < inSize; j++) {
        acc += weights[i * inSize + j] * input[j];
    }
    output[i] = 1.0 / (1.0 + exp(-acc));
}
