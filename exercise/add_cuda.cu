#include<cuda_runtime.h>
#include<vector>
#include<iostream>
const size_t SIZE = 1 << 20;
size_t size_bytes = SIZE * sizeof(float);


dim3 block_dim(256);
dim3 grid_dim((SIZE + block_dim.x-1)/block_dim.x);

template<typename T>
__global__ void add_kernel(T *c, T *a, T *b, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    // 数据初始化
    std::vector<float> h_a(SIZE,1);
    std::vector<float> h_b(SIZE,2);
    std::vector<float> h_c(SIZE,0);

    // 分配device端memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_bytes);
    cudaMalloc(&d_b, size_bytes);
    cudaMalloc(&d_c, size_bytes);

    // 将数据从host端拷贝到device端
    cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice);

    add_kernel<<<grid_dim,block_dim>>>(d_c, d_a, d_b, SIZE);

    cudaMemcpy(h_c.data(),d_c,size_bytes,cudaMemcpyDeviceToHost);
    for(auto i : h_c) std::cout<<i<<' ';
    if(d_a) cudaFree(d_a);
    if(d_b) cudaFree(d_b);
    if(d_c) cudaFree(d_c);
    
}