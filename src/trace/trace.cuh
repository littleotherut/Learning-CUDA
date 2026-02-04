#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__device__ T warp_reduce(T val){
    #pragma unroll
    for(size_t i = warpSize/2; i > 0 ; i>>=1){
        val += __shfl_down_sync(0xffffffff,val,i);
    }
    return val;
}

template<typename T>
__global__ void reduce_smem_warp_trace_kernel(T *result, const T *input, size_t n, size_t cols){
    extern __shared__ unsigned char smem_bytes[];
    T *smem = reinterpret_cast<T*>(smem_bytes);
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    T sum = 0;
    for(size_t i = idx ; i < n ; i += blockDim.x * gridDim.x){
        size_t i_idx = i * (cols + 1); // 计算对角线元素的索引
        sum += input[i_idx];
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
void trace(T *result, const T *input, size_t n, size_t cols,dim3 block_dim = dim3(256) ){
    if (n == 0) return;
    dim3 grid_dim = (n + block_dim.x - 1 ) / block_dim.x;
    // 计算共享内存大小：每个 warp 需要一个 T 大小的空间
    int num_warps = (block_dim.x + 32 - 1) / 32;
    size_t smem_size = num_warps * sizeof(T);
    // atomic_sum_kernel<<<grid_dim, block_dim>>>(result,input,n);
    reduce_smem_warp_trace_kernel<<<grid_dim, block_dim, smem_size>>>(result,input,n, cols);
    RUNTIME_CHECK(cudaGetLastError());
}
