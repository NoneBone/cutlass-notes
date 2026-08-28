# 06 · Block Copy：用共享内存缓存 block tile

本目录在 `05-block-mma` 的 block-level MMA 之上加入共享内存（shared memory, SMEM）缓存，学习一个 GEMM block 的典型数据流：

```text
gmem --(cp.async, G2S)--> smem --(S2R)--> registers --(MMA)--> registers
  --(R2S)--> smem --(S2G)--> gmem
```

示例固定计算 `128×128×64` tile：A、B 是 BF16，C 在 FP32 中作为累加器输入，最终 O 为 BF16。相比上一节，A/B/C 不再直接从 gmem 进入寄存器，而是先集中搬入 SMEM，再按 MMA 所需的布局读取。

SMEM 的价值是让 block 内多个 warp 复用 tile 数据，并把全局内存访问与 Tensor Core 需要的寄存器布局解耦。本例尚未做多 stage 流水线，重点是理解内存层次、异步复制和同步时机。

## 文件导览

| 文件 | 作用 | 建议 |
| --- | --- | --- |
| `block_copy.py` | 编译 C++ 扩展、创建输入并验证 MM/MMA | 先读 |
| `block_copy.cu` | C++ 版 SMEM 分配、cp.async、S2R/R2S/S2G copy | 重点精读 |
| `cutedsl_block_copy.py` | CuTe DSL 版，显式构造 G2S、S2R、R2S、S2G TiledCopy | 对照 C++ |
| `profile.sh` | Nsight Compute / Nsight Systems 命令 | 最后尝试 |

需要 SM80 或更高的 CUDA GPU、PyTorch CUDA 扩展编译环境和匹配的 CuTe DSL。运行：

```bash
cd 06-block-copy
python block_copy.py
python cutedsl_block_copy.py
```

当前是单 `(1,1,1)` block 网格，没有边界残块；动态 SMEM 约 128 KB，实际可运行性取决于 GPU 的 SMEM 上限。

## `block_copy.py`：Python 端重点

### 1. 计算结果与上一节一致

```python
M, N, K = 128, 128, 64
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
c = torch.randn(M, N, device="cuda", dtype=torch.float32)

out = lib.block_copy_bf16_bf16_bf16_fp32(a, b, c)
ref = torch.addmm(c, a.float(), b.T.float()).bfloat16()
```

`TiledCopy` 改变的是数据搬运路径，不改变 GEMM 数学。参考实现仍要先用 FP32 完成乘加，最后转换为 BF16。传入 `None` 是清零累加器的 MM；传入 C 是预载 C 的 MMA。

### 2. 用 `ENABLE_PROF` 避免验证干扰性能分析

脚本在 `ENABLE_PROF` 为真时跳过 PyTorch 参考和误差打印，只保留 kernel 执行。这种“验证模式 / profiling 模式”分离是编写微基准时的常见做法；性能分析时可减少额外 kernel 和同步对结果的影响。

## `cutedsl_block_copy.py`：值得学习的代码段

### 1. 128-bit `cp.async` 的 G2S 布局

```python
g2s_op = cute.nvgpu.cpasync.CopyG2SOp(
    cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
)
g2s_atom_a = cute.make_copy_atom(g2s_op, mA.element_type, num_bits_per_copy=128)
g2s_tiled_copy_a = cute.make_tiled_copy_tv(g2s_atom_a, tlAB_thr, tlAB_val)
```

每次复制是 128 bit，即 16 字节。BF16 每元素 2 字节，所以一个线程一次搬 8 个元素；FP32 C 每元素 4 字节，一次搬 4 个元素。`tlAB_thr` 描述线程如何沿 K 方向分工，`tlAB_val` 描述单线程一次复制的值形状。

### 2. cp.async 提交、等待和 block 同步

```python
cute.copy(g2s_tiled_copy_a, tAgA_g2s, tAsA_g2s)
cute.copy(g2s_tiled_copy_b, tBgB_g2s, tBsB_g2s)
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()
```

`cute.copy` 发出异步 gmem→smem 复制；`commit_group` 将此前的异步复制归组，`wait_group(0)` 等待全部未完成组，`sync_threads()` 再确保整个 block 都能看到 SMEM 数据。少掉任一层同步，都可能导致读到尚未完成的数据。

### 3. SMEM→寄存器必须匹配 MMA fragment

```python
s2r_tiled_copy_a = cute.make_tiled_copy_A(s2r_atom_a, tm)
thr_s2r_a = s2r_tiled_copy_a.get_slice(tid)
tAsA_s2r = thr_s2r_a.partition_S(sA)
tArA_s2r = thr_s2r_a.retile(tCrA)
cute.copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r)
```

G2S 追求合并且宽的全局访问；S2R 则追求与 Tensor Core fragment 相容。两者使用不同 TiledCopy 是正常的。A/B/C 分别由 `make_tiled_copy_A/B/C` 从同一个 `TiledMma` 推导布局，避免手写线程索引。

### 4. R2S→S2G 的 epilogue

```python
tCrO = cute.make_fragment_like(tCrC, out_dtype)
tCrO.store(tCrC.load().to(out_dtype))
# R2S 写入 sO，随后 S2G 写入 gO
```

先在寄存器中把 FP32 累加器转换为 BF16，再写入 SMEM 输出缓冲区，最后由另一套 S2G copy 连续写回 gmem。中间的 SMEM 缓冲区让寄存器 fragment 布局和全局连续写布局可以分别优化。

## `block_copy.cu`：C++ 对照重点

### 1. 动态 SMEM 的分区

```cpp
extern __shared__ __align__(1024) uint8_t smem[];
uint8_t *Aptr_smem = smem;
uint8_t *Bptr_smem = smem + kShmSizeA;
uint8_t *Cptr_smem = smem + kShmSizeA + kShmSizeB;
```

kernel 使用一段动态字节数组，再按编译期计算的大小切分成 A、B、C、O 缓冲区。`kShmSize` 决定 launch 时申请多少动态 SMEM。理解字节偏移和元素类型，是排查 SMEM 越界的基础。

### 2. 为什么 MMA 后还要 `__syncthreads()`

MMA 结果先由各线程写入 SMEM 的 R2S 阶段，随后其他线程可能从同一 SMEM 缓冲区执行 S2G。R2S 完成后必须 block 同步，保证所有线程写完再开始读取；这也是代码中注释“Why should we sync here?”的答案。

### 3. `Copy_S2R_op` 的类型选择

C++ 根据元素大小选择 `ldmatrix` 或 `AutoVectorizingCopy`：16-bit 的 A/B 可使用 `SM75_U32x4_LDSM_N`，而 FP32 C 不满足 ldmatrix 的 16-bit 限制，只能使用通用复制。数据类型会直接影响可用的硬件搬运指令。

## 小作业

1. 运行两个脚本，记录 kernel 输出 dtype、相对误差和 block 的动态 SMEM 大小。
2. 计算一次 128-bit G2S copy 对 BF16 和 FP32 分别搬运多少元素；结合 256 个线程估算一次 A tile 的复制次数。
3. 暂时注释 `cp_async_wait_group(0)` 或 `sync_threads()`，观察结果或编译行为，并解释为什么异步复制不能立即当作完成。
4. 把 A、B 设为全 1、C 设为全 2。由于 K=64，预测 MM 输出 64，MMA 输出 66，最后 BF16 结果应接近这些值。
5. 思考题：若将本例扩展为两个 K stage，哪些 SMEM 缓冲区需要变成三维（增加 stage 维），同步和 wait/commit 顺序应如何安排？

完成后可继续阅读 `07-swizzling`，观察 SMEM 布局如何进一步解决 `ldmatrix` 的 bank conflict。
