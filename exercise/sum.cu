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

// 最基础；原子加法
template <typename T>
__global__ void atomic_sum_kernel(T* result, const T* input, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        atomicAdd(result,input[idx]);
        // * result += input[idx];
    }
}
// 对warp进行规约（未使用共享内存或shf
template <typename T>
__global__ void reduce_warp_sum_kernel_old(T *result, const T *input, size_t n){
    
    const int warp_size = warpSize;
    size_t lane = threadIdx.x % warp_size;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (lane == 0){
        T warp_sum = 0;
        for(size_t i = 0 ; i < warp_size ; ++ i){
            for(size_t idx_t = idx + i ; idx_t < n ; idx_t += gridDim.x * blockDim.x){
                warp_sum += input[idx_t];
            }
        }
        atomicAdd(result,warp_sum);
    }
}

// 采用shfl，减少空闲，寄存器操作高效
template<typename T>
__device__ T warp_reduce(T val){
    #pragma unroll
    for(size_t i = warpSize/2; i > 0 ; i>>=1){
        val += __shfl_down_sync(0xffffffff,val,i);
    }
    return val;
}
template<typename T>
__global__ void reduce_warp_sum_kernel(T *result, const T *input, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    T sum = 0;
    for(size_t i = idx ; i < n ; i += blockDim.x * gridDim.x){
        sum += input[i];
    }
    sum = warp_reduce(sum);
    if(threadIdx.x % warpSize==0)
        atomicAdd(result,sum);
}

// 采用共享内存，原子操作个数压缩到block个数。
template<typename T>
__global__ void reduce_smem_sum_kernel(T *result, const T *input, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = idx < n ? input[idx] : 0;
    __syncthreads();

    for(size_t i = blockDim.x / 2 ; i > 0 ; i >>=1 ){
        if(tid < i) smem[tid] += smem[tid+i];
        __syncthreads();
    }

    if(tid == 0) atomicAdd(result,smem[0]);
}

template<typename T>
__global__ void reduce_smem_warp_sum_kernel(T *result, const T *input, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    T sum = 0;
    for(size_t i = idx ; i < n ; i += blockDim.x * gridDim.x){
        sum += input[i];
    }
    sum = warp_reduce(sum);

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    if(lane == 0){
        smem[warp_id] = sum;
    }
    __syncthreads();
    
    if(tid < warpSize){
        int num = (blockDim.x + warpSize - 1)/ warpSize ;
        T block_sum = (tid < num) ? smem[tid] : 0;
        block_sum = warp_reduce(block_sum);
        if(tid==0) atomicAdd(result,block_sum);
    }
}

template<typename T>
void sum(T *result, const T *input, size_t n, dim3 block_dim = (256)){
    dim3 grid_dim = (n + block_dim.x - 1 ) / block_dim.x;
    // 计算共享内存大小：每个 warp 需要一个 T 大小的空间
    int num_warps = (block_dim.x + 32 - 1) / 32;
    size_t smem_size = num_warps * sizeof(T);
    // atomic_sum_kernel<<<grid_dim, block_dim>>>(result,input,n);
    reduce_smem_warp_sum_kernel<<<grid_dim, block_dim, smem_size>>>(result,input,n);
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