#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

template<typename T>
struct TypeTraits{
    using host_type = T;
    static T from_float(float val) {
        return static_cast<T>(val);
    }
    static float to_float(T val) {
        return static_cast<float>(val);
    }
};
template<>
struct TypeTraits<half>{
    using host_type = float;
    static half from_float(float val) {
        return __float2half(val);
    }
    static float to_float(half val) {
        return __half2float(val);
    }
};
template<>
struct TypeTraits<float2>{
    using host_type = float2;
    static float2 from_float(float val) {
        return make_float2(val, val);
    }
    static float to_float(float2 val) {
        return (val.x + val.y) / 2.0f;
    }
};
template<>
struct TypeTraits<float4>{
    using host_type = float4;
    static float4 from_float(float val) {
        return make_float4(val, val, val, val);
    }
    static float to_float(float4 val) {
        return (val.x + val.y + val.z + val.w) / 4.0f;
    }
};

// 模板特化的加法操作
template<typename T>
__device__ T add(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, half>) {
        return __hadd(a, b);
    }
    else if constexpr (std::is_same_v<T, float2>) {
        return make_float2(a.x + b.x, a.y + b.y);
    }
    else if constexpr (std::is_same_v<T, float4>) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    else {
        return a + b;
    }
}

// Kernel 函数
template<typename T>
__global__ void add_kernel(T* c, const T* a, const T* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        c[i] = add(a[i], b[i]);
    }
}

// Host 函数封装
template<typename T>
void vector_add(T* d_c, const T* d_a, const T* d_b, size_t n, 
                dim3 block_dim = (256)) {
    dim3 grid_dim = (n + block_dim.x - 1) / block_dim.x;
    add_kernel<<<grid_dim, block_dim>>>(d_c, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
}
using type_t = float2;
int main() {
    constexpr size_t SIZE = 1 << 20;
    constexpr size_t SIZE_BYTES = SIZE * sizeof(type_t);
    
    // Host 数据初始化
    std::vector<type_t> h_a(SIZE, TypeTraits<type_t>::from_float(1.2f));
    std::vector<type_t> h_b(SIZE, TypeTraits<type_t>::from_float(2.2f));
    std::vector<type_t> h_c(SIZE);

    // 分配 device 内存
    type_t *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, SIZE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_b, SIZE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_c, SIZE_BYTES));

    // Host to Device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), SIZE_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), SIZE_BYTES, cudaMemcpyHostToDevice));

    // 执行 kernel
    vector_add(d_c, d_a, d_b, SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Device to Host
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, SIZE_BYTES, cudaMemcpyDeviceToHost));

    // 验证结果 (只打印前 10 个和最后 10 个)
    std::cout << "First 10 results: ";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << TypeTraits<type_t>::to_float(h_c[i]) << " ";
    }
    std::cout << "\nLast 10 results: ";
    for (size_t i = SIZE - 10; i < SIZE; ++i) {
        std::cout << TypeTraits<type_t>::to_float(h_c[i]) << " ";
    }
    std::cout << std::endl;

    // 清理资源
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}