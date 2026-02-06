#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
#include "trace/trace.cuh"
#include "attention/attention.cuh"
/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  size_t SIZE = rows < cols ? rows : cols;
  size_t SIZE_BYTES = sizeof(T) * rows * cols;
  T result_h = T(0);

  T *result_d, *input_d;
  RUNTIME_CHECK(cudaMalloc(&input_d, SIZE_BYTES));
  RUNTIME_CHECK(cudaMalloc(&result_d, sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(input_d,h_input.data(),SIZE_BYTES,cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(result_d,&result_h,sizeof(T),cudaMemcpyHostToDevice));

  trace(result_d,input_d,SIZE, cols);
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(&result_h,result_d, sizeof(T), cudaMemcpyDeviceToHost));

  // std::cout<< result_h << std::endl;

  RUNTIME_CHECK(cudaFree(input_d));
  RUNTIME_CHECK(cudaFree(result_d));
  return result_h;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // 计算各张量的元素数量
  size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
  size_t kv_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
  size_t o_size = q_size;  // 输出与 Q 同形状

  // 在 GPU 上分配内存
  T *d_q, *d_k, *d_v, *d_o;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));

  // 将输入数据从 Host 拷贝到 Device
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));

  // 启动 Flash Attention kernel (自动根据 head_dim 选择模板特化版本)
  launch_flash_attention<T>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim,
      is_causal);

  RUNTIME_CHECK(cudaDeviceSynchronize());

  // 将结果从 Device 拷贝回 Host
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));

  // 释放 GPU 内存
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
