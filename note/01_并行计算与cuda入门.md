# 并行计算与CUDA入门
## 区别
- CPU并行：单体已具备强实力，处理复杂计算与逻辑处理，高同步
- GPU并行：单体较弱，大规模数据的简单并行计算，高吞吐
## CUDA编程
- C/C++语法、SIMT（一指令，多线程执行）、CPU协作、自动调度
## GPU单元
- CUDA Core/SP：负责通用计算
## 运算过程
1. CPU准备数据，存储在RAM主存；
2. 通过Bus/总线传输给Global Memory（GPU）
3. GPU从GM读数据，运算，并写回；
4. 总线传输回CPU
## 线程编号
SIMT指挥每个线程，需要组织结构和编号：idx = BlockId * BlockSize + ThreadId
- Grid -> Block -> Thread
grid、block可以有三维，设置grid具备的block个数时，采用$ 运算总数 / 线程数 $ 上取整，同时核心函数需要有越界判断。
## 瓶颈
### 性能分析：`nsys`(Nsight Systems):
- 启动profiling: `nsys profile -t cuda,nvtx,osrt -o add_cuda -f true ./add_cuda`
- 解析并统计性能信息：`nsys stats add_cuda.nsys-rep`
1. 核函数唤醒(可修改实现解决)
2. 通过总线进行写入与写回的开销
# 性能模型与逐元素优化
问题： HtoD 和 DtoH 的耗时远大于核函数时间