/* Dense Forward kernel — matmul + bias + sigmoid activation.
 * One work-item per output neuron i.
 *
 * weights[i * inSize + j] for j in [0, inSize]
 * bias[i]
 * input[j]  for j in [0, inSize]
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

/* Dense Backward kernels — COMP-3 (parity vs CPU within 1e-4).
 *
 * dense_grad_w: ∂L/∂W[i,j] = dY[i] * X[j] — outer product, one work-item per (i,j).
 *   Global size: outSize * inSize.
 *
 * dense_grad_x: ∂L/∂X[j] = Σ_i W[i,j] * dY[i] — one work-item per j.
 *   Global size: inSize.
 */

__kernel void dense_grad_w_f32(
    __global const float* dY,
    __global const float* input,
    __global       float* gradW,
    const int inSize,
    const int outSize
) {
    int idx = get_global_id(0); /* flattened: i * inSize + j */
    int total = outSize * inSize;
    if (idx >= total) return;
    int i = idx / inSize;
    int j = idx % inSize;
    gradW[idx] = dY[i] * input[j];
}

__kernel void dense_grad_x_f32(
    __global const float* weights,
    __global const float* dY,
    __global       float* gradX,
    const int inSize,
    const int outSize
) {
    int j = get_global_id(0);
    if (j >= inSize) return;
    float acc = 0.0f;
    for (int i = 0; i < outSize; i++) {
        acc += weights[i * inSize + j] * dY[i];
    }
    gradX[j] = acc;
}

__kernel void dense_grad_w_f64(
    __global const double* dY,
    __global const double* input,
    __global       double* gradW,
    const int inSize,
    const int outSize
) {
    int idx = get_global_id(0);
    int total = outSize * inSize;
    if (idx >= total) return;
    int i = idx / inSize;
    int j = idx % inSize;
    gradW[idx] = dY[i] * input[j];
}

__kernel void dense_grad_x_f64(
    __global const double* weights,
    __global const double* dY,
    __global       double* gradX,
    const int inSize,
    const int outSize
) {
    int j = get_global_id(0);
    if (j >= inSize) return;
    double acc = 0.0;
    for (int i = 0; i < outSize; i++) {
        acc += weights[i * inSize + j] * dY[i];
    }
    gradX[j] = acc;
}
