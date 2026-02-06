#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// T -> float & float -> T
template <typename T>
__device__ __forceinline__ float to_float(T val) { return (float)val; }

template <>
__device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }

template <typename T>
__device__ __forceinline__ T from_float(float val) { return (T)val; }

template <>
__device__ __forceinline__ half from_float<half>(float val) { return __float2half(val); }

/*
flashAttention
基础attention ：
Output = softmax(Q K^T / sqrt(d_k)) V
Q: [batch_size, target_seq_len, query_head, head_dim]
K: [batch_size, src_seq_len, kv_heads, head_dim]
V: [batch_size, src_seq_len, kv_heads, head_dim]
Output: [batch_size, target_seq_len, query_head, head_dim]

flash attention优化：
计算Q K^T时，直接计算每个block的结果，并且在计算过程中进行softmax归一化，避免了中间结果的存储和读取。
分块策略：
每个block处理(b,h,q_i : q_i + BLOCK_M)
每个线程处理一个head_dim维度的数据。
*/
template <typename T, int MAX_HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_kernel(
    const T* __restrict__ Q, const T* __restrict__ K, 
    const T* __restrict__ V, T* __restrict__ O, 
    int batch_size, int target_seq_len, int src_seq_len,
    int query_head, int kv_heads, int head_dim, bool is_causal) {

    const int q_block_start = blockIdx.x * BLOCK_M;
    const int bh = blockIdx.y;
    const int batch_id = bh / query_head;
    const int head_id = bh % query_head;
    const int tid = threadIdx.x;

    const int heads_per_group = query_head / kv_heads; // GQA 分组数量
    const int kv_h = head_id / heads_per_group; // K/V头索引

    const int i = q_block_start + tid;
    const bool valid = (i < target_seq_len);

    const float scale = rsqrtf((float)head_dim);

    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + BLOCK_N * head_dim;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    float o_acc[MAX_HEAD_DIM];
    for(int d = 0; d < MAX_HEAD_DIM; d++) {
        o_acc[d] = 0.0f;
    }

    const int q_base = valid ?
        ((batch_id * target_seq_len + i) * query_head + head_id) * head_dim : 0;

    const int num_kv_tiles = (src_seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int tile = 0; tile < num_kv_tiles; ++ tile) {
        const int tile_start = tile * BLOCK_N;
        if(is_causal && tile_start > q_block_start + BLOCK_M - 1) break;

        for(int load_idx = tid; load_idx < BLOCK_N * head_dim; load_idx += BLOCK_M) {
            const int row = load_idx / head_dim;
            const int col = load_idx % head_dim;
            const int j = tile_start + row;
            if(j < src_seq_len) {
                const int offset = ((batch_id * src_seq_len + j) * kv_heads + kv_h) * head_dim + col;
                K_smem[load_idx] = to_float(K[offset]);
                V_smem[load_idx] = to_float(V[offset]);
            } else {
                K_smem[load_idx] = 0.0f;
                V_smem[load_idx] = 0.0f;
            }
        }
        __syncthreads();

        if(valid && !(is_causal && tile_start > i)) {
            const int tile_end = min(tile_start + BLOCK_N, src_seq_len);
            const int tile_len = tile_end - tile_start;

            // 计算 Q 与 K 的点积，并进行 softmax 归一化，同时累积加权的 V,
            for(int j = 0; j < tile_len; ++ j){
                if (is_causal && (tile_start + j) > i) break;

                float dot = 0.0f;
                for(int d = 0; d < head_dim; ++ d) {
                    dot += to_float(Q[q_base + d]) * K_smem[j * head_dim + d];
                }
                const float score = dot * scale;
                // 在线softmax
                // 公式： m_ij = max(m_i, score)
                //       l_i = l_i * exp(m_i - m_ij) + exp(score - m_ij)
                //       o_acc = o_acc * exp(m_i - m_ij) + exp(score - m_ij) * V_smem[j]
                const float m_ij = fmaxf(m_i, score);
                const float correction = expf(m_i - m_ij);
                const float p = expf(score - m_ij);
                l_i = l_i * correction + p;
                for(int d = 0; d < head_dim; ++ d) {
                    o_acc[d] = o_acc[d] * correction + p * V_smem[j * head_dim + d];
                }
                m_i = m_ij;
            }
        }
        __syncthreads();
    }
    
    if (valid && l_i > 0.0f) {
        const int o_base = ((batch_id * target_seq_len + i) * query_head + head_id) * head_dim;
        const float inv_l = 1.0f / l_i;
        for (int d = 0; d < head_dim; d++) {
            O[o_base + d] = from_float<T>(o_acc[d] * inv_l);
        }
    }

}


template <typename T>
void launch_flash_attention(
    const T* d_q, const T* d_k, const T* d_v, T* d_o,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    constexpr int BM = 64;   // 每个线程块处理 64 个 query 位置
    constexpr int BN = 32;   // K/V 分块大小

    // 2D grid: x 轴 = query 块索引, y 轴 = batch * head
    dim3 grid((target_seq_len + BM - 1) / BM, batch_size * query_heads);
    dim3 block(BM);

    constexpr int MAX_HEAD_DIM = 256;
    size_t smem = 2 * BN * head_dim * sizeof(float);
    flash_attention_kernel<T, MAX_HEAD_DIM, BM, BN><<<grid, block, smem>>>(
        d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);

}