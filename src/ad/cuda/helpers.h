#pragma once

#include <cublas_v2.h>
#include <memory>
#include <iostream>

#define CUDA_CALL(XXX) \
    do { auto err = XXX; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << \
        cudaGetErrorString(err) << ", at line " << __LINE__ \
        << std::endl; std::terminate(); } /*cudaDeviceSynchronize();*/} while (0)

namespace cuda {

class CublasHandle {
    cublasHandle_t cuhandle_;

    public:
        cublasHandle_t get() const { return cuhandle_; }

        CublasHandle() {
            cublasCreate(&cuhandle_);
        }

        ~CublasHandle() {
            cublasDestroy(cuhandle_);
        }
};

thread_local extern CublasHandle g_cuhandle;

} // cuda

namespace cuda {
namespace helpers {

struct CUFree {
    template <class T>
    void operator()(T* ptr) const { CUDA_CALL(cudaFree(ptr)); }
};

template <class T>
using cunique_ptr = std::unique_ptr<T, CUFree>;

template <class T>
T* cunew(size_t n) {
    T* ptr;
    CUDA_CALL(cudaMalloc((void**)&ptr, sizeof (T) * n));
    return ptr;
}

template <class T>
void CPUToGPU(T* dst, const T* src, size_t n) {
    CUDA_CALL(cudaMemcpy(
            dst, src,
            sizeof(T) * n,
            cudaMemcpyHostToDevice));
}

template <class T>
void GPUToGPU(T* dst, const T* src, size_t n) {
    CUDA_CALL(cudaMemcpy(
            dst, src,
            sizeof(T) * n,
            cudaMemcpyDeviceToDevice));
}

template <class T>
void GPUToCPU(T* dst, const T* src, size_t n) {
    CUDA_CALL(cudaMemcpy(
            dst, src,
            sizeof(T) * n,
            cudaMemcpyDeviceToHost));
}

void fill(float* array, size_t size, float val);

} // helpers
} // cuda
