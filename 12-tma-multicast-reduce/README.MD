# 12 · TMA Multicast/Reduce：集群内复用加载并合并写回

本目录在 11 的 TMA load/store 之上使用 Hopper thread-block cluster（默认 `Shape<_2,_1,_1>`）：两个 CTA 沿 cluster.x 协作。A tile 对两个 CTA 相同，因此只加载一次并 multicast；两个 CTA 分别计算不同的 N tile。对于 C 与输出类型相同的配置，还可用 TMA reduce-add 将各 CTA 的结果直接累加到全局 C。

```text
                 ┌─ CTA (x=0) ─ MMA ─┐
gmem A ─TMA mcast┤                   ├─ TMA reduce-add → C
                 └─ CTA (x=1) ─ MMA ─┘
gmem B ─TMA load（每个 CTA 各自的 N tile）
```

## 文件导览与运行

| 文件 | 作用 |
| --- | --- |
| `tma_multicast_reduce.py` | 编译扩展、测试三种 dtype 配置 |
| `tma_multicast_reduce.cu` | cluster launch、multicast mask、reduce-add 实现 |
| `cutedsl_tma_multicast_reduce.py` | CuTe DSL 对照实现，展示 cluster layout 与 pipeline |
| `profile.sh` | Nsight Compute 分析入口 |

```bash
cd 12-tma-multicast-reduce
python tma_multicast_reduce.py
python cutedsl_tma_multicast_reduce.py
```

需要支持 SM90 cluster/TMA 的 Hopper GPU。示例尺寸偏大（M/N 从 256 起），初学时可先缩小列表验证功能。

## Python 端值得学习的部分

### 1. 用两种调用验证 reduce-add 语义

```python
o1 = lib.tma_multicast_reduce_fp16_fp16_fp16_fp16(a, b, None) + c
o2 = lib.tma_multicast_reduce_fp16_fp16_fp16_fp16(a, b, c)
compare_matrix(o1, o2)
```

`c=None` 时 C++ harness 创建零张量，返回 `A @ B.T`；提供 C 时，reduce-add 路径把结果加到用户的 C。两种调用相等，正好验证“全局内存归约加法”的效果。

### 2. 区分精度路径

FP16 累加且输出 FP16 时可使用 `SM90_TMA_REDUCE_ADD`；FP32 accumulator、FP16/BF16 output 时需要先加载 C、在寄存器中相加，再普通 TMA store。输出类型与 accumulator 不同是选择路径的重要条件。

## `tma_multicast_reduce.cu`：建议精读的代码段

### 1. cluster 坐标与 multicast mask

```cpp
auto cta_layout_mnk = make_layout(typename Spec::ClusterShape{});
auto cta_coord_mnk = cta_layout_mnk.get_flat_coord(block_rank_in_cluster);
uint16_t mcast_mask_a = create_tma_multicast_mask<0>(cta_layout_mnk, cta_coord_mnk);
uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_mnk, cta_coord_mnk);
```

`block_rank_in_cluster` 把当前 CTA 映射到 cluster 坐标。cluster.x=2 时 A 的 mask 覆盖两个 CTA；cluster.y=1 时 B 不需要 multicast，mask 只覆盖本 CTA。改变 cluster shape 后，必须重新思考 tile 复用方向、mask 和 grid 对齐。

### 2. 带 multicast 的 TMA partition 与加载

```cpp
auto [tAgA, tAsA] = tma_partition(tma_A, get<0>(cta_coord_mnk),
                                   make_layout(size<0>(cta_layout_mnk)), ...);
copy(tma_A.with(tma_load_mbarrier, mcast_mask_a), tAgA(_, ik), tAsA);
```

TMA descriptor 通过 cluster 维度知道一个 tile 要发送给哪些 CTA；同一个发起线程配合 multicast mask，把 A 搬运结果广播到伙伴 CTA 的 SMEM。每个消费者仍必须正确参与 barrier 到达计数。

### 3. cluster 同步与 barrier 生命周期

```cpp
if constexpr (Spec::kClusterSize > 1) cluster_sync();
else __syncthreads();
...
invalidate_barrier(tma_load_mbarrier);
```

cluster 内共享的 TMA 操作需要 `cluster_sync()`，普通 block 内数据交换才使用 `__syncthreads()`。barrier 初始化、每个 K tile 的 phase 等待、最终 invalidate 都必须在所有 CTA 的一致控制流中完成，否则容易出现 launch failure 或死等。

### 4. TMA reduce-add 与普通 store 的分支

```cpp
if constexpr (use_tma_store_reduce_add) {
  copy(r2s_tiled_copy_d, tCrC_r2s, tCsC_r2s);
  tma_store_fence();
  copy(tma_C, tCsC, tCgC);  // SM90_TMA_REDUCE_ADD
} else {
  // TMA load C → S2R → accumulator += C → TMA store D
}
```

`if_use_tma_store_reduce_add` 在编译期判断 `ComputeTypeC == OutType`。满足条件时，TMA 的 reduction op 在全局内存端完成加法；不满足时不能把 FP32 accumulator 直接 reduce 到较窄类型，因此走显式 C load 和类型转换路径。

### 5. cluster launcher

```cpp
cutlass::launch_kernel_on_cluster({grid, block, cluster, shm_size, stream},
                                  kernel_ptr, tma_A, tma_B, tma_C, tma_D, M, N, K);
```

cluster kernel 不能简单依赖普通 `<<<grid, block>>>` 语义；示例用 CUTLASS launcher 传递 cluster 维度、动态 SMEM 和 TMA descriptor。调试时先确认 GPU、CUDA 版本和 cluster launch 支持，再排查 kernel 内逻辑。

## 小作业

1. 把 `ClusterShape` 改为 `Shape<_1,_1,_1>`，比较 mask、同步调用和结果，理解 multicast 关闭后的退化情况。
2. 尝试 `Shape<_2,_2,_1>`：判断 A、B 各自应沿哪个 cluster 方向 multicast，并估算需要的 mask/同步变化。
3. 将 FP16 的 `c` 初始化为全 1，验证 `c + A @ B.T` 与 reduce-add 输出一致。
4. 在 kernel 中记录每个 CTA 的 `block_rank_in_cluster` 与 `cta_coord_mnk`，画出两个 CTA 对应的 M/N tile。
5. 解释为什么 FP32 accumulator→FP16 output 默认不直接使用 reduce-add；指出需要增加哪些显式转换或临时缓冲才可能支持它。

