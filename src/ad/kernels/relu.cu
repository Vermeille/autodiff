#include "kernels.h"

namespace ad {
namespace cuda {

__global__
void cuRelu(float* res, const float* arr1, size_t sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        res[i] = (arr1[i] > 0) ? arr1[i] : 0;
    }
}

void Relu(float* res, const float* arr1, size_t sz) {
    cuRelu<<<(sz + 128 - 1) / 128, 128>>>(res, arr1, sz);
    cudaDeviceSynchronize();
}

__global__
void cuReluGrad(float* res, const float* arr1, size_t sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        res[i] = (arr1[i] > 0) ? 1 : 0;
    }
}

void ReluGrad(float* res, const float* arr1, size_t sz) {
    cuReluGrad<<<(sz + 128 - 1) / 128, 128>>>(res, arr1, sz);
    cudaDeviceSynchronize();
}

} // cuda
} // ad
