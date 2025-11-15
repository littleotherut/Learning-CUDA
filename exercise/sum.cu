#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>
#include <iostream>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

template <typename T>
__global__ void atomic_sum_kernel(T* result, const T* input, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        atomicAdd(result,input[idx]);
        // * result += input[idx];
    }
}

template<typename T>
void sum(T *result, const T *input, size_t n, dim3 block_dim = (256)){
    dim3 grid_dim = (n + block_dim.x - 1 ) / block_dim.x;
    atomic_sum_kernel<<<grid_dim, block_dim>>>(result,input,n);
    CUDA_CHECK(cudaGetLastError());
}

using type_t = int;
int main(){
    constexpr size_t SIZE = 1 << 20;
    constexpr size_t SIZE_BYTES = SIZE * sizeof(type_t);
    std::vector<type_t> input_h(SIZE, 1.0f);
    type_t result_h = 0;

    type_t *input_d, *result_d;
    CUDA_CHECK(cudaMalloc(&input_d, SIZE_BYTES));
    CUDA_CHECK(cudaMalloc(&result_d, sizeof(type_t)));

    CUDA_CHECK(cudaMemcpy(input_d,input_h.data(),SIZE_BYTES,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result_d,&result_h,sizeof(type_t),cudaMemcpyHostToDevice));

    sum(result_d,input_d,SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result_h,result_d, sizeof(type_t), cudaMemcpyDeviceToHost));

    std::cout<< result_h << std::endl;

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(result_d));

    return 0;
}