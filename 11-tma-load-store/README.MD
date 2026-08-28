# 11 · TMA Load/Store：让 Tensor Memory Accelerator 搬运 tile

本目录把 `09-pipelining` 的 `cp.async` 全局↔共享内存搬运替换为 Hopper（SM90）Tensor Memory Accelerator（TMA）。TMA 根据预先创建的 tensor map，一次由少数线程发起整个二维 tile 的搬运，其余线程等待 mbarrier 后直接从 SMEM 做 MMA；输出也由 TMA 写回全局内存。

```text
host: tensor + layout --make_tma_copy--> TMA descriptor
device: TMA load(gmem→smem) --mbarrier--> S2R/ldmatrix → MMA
        MMA → R2S --tma_store_fence--> TMA store(smem→gmem)
```

## 文件导览与运行

| 文件 | 作用 |
| --- | --- |
| `tma_load_store.py` | 编译扩展、遍历尺寸、与 PyTorch 结果比较 |
| `tma_load_store.cu` | TMA descriptor、mbarrier、load/store 与 MMA 主循环 |
| `cutedsl_tma_load_store.py` | CuTe DSL 对照实现 |
| `profile.sh` | Nsight Compute 分析入口 |

```bash
cd 11-tma-load-store
python tma_load_store.py
python cutedsl_tma_load_store.py
```

需要支持 TMA 的 SM90/Hopper GPU。示例中的 N、K 仍需是 8 的倍数；初学时可把尺寸列表缩成 `M=130,N=136,K=64`，观察边界行为。

## Python 端值得学习的部分

### 1. TMA 版本保持相同的数值验证接口

```python
kernel_output = lib.tma_load_store_fp16_fp16_fp16_fp32(a, b, None)
torch_output = torch.matmul(a, b.T)
compare_matrix(kernel_output, torch_output)
```

从 Python 看，TMA kernel 与前几节的函数签名几乎相同；性能优化主要封装在 C++ descriptor 和 device kernel 中。先确认 dtype、转置关系与累加精度，再比较性能。

### 2. `ENABLE_PROF` 与 `ENABLE_MMA`

`ENABLE_PROF=1` 时跳过逐元素比较，避免 profiling 被打印和同步干扰；`ENABLE_MMA=1` 才测试提供 C 的 `C + A @ B.T` 路径。建议先用小尺寸验证，再打开 profiling。

## `tma_load_store.cu`：建议精读的代码段

### 1. 创建 TMA descriptor 与分区

```cpp
auto mA = make_tensor(make_gmem_ptr((ComputeTypeA*)a.data_ptr()),
                      make_layout(make_shape(M, K), LayoutRight{}));
auto tma_A = make_tma_copy(SM90_TMA_LOAD{}, mA, smem_layout_A);
auto [tAgA, tAsA] = tma_partition(
    tma_A, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));
```

`make_tma_copy` 把全局 tensor 的布局和目标 SMEM tile 编码进 descriptor；`tma_partition` 返回可供 TMA 发起者使用的 global tile (`tAgA`) 和 SMEM 目标 (`tAsA`)。这一步决定了 TMA 是否满足连续维度、对齐和 tile 形状约束。

### 2. mbarrier 配置、发起和等待

```cpp
initialize_barrier(tma_load_mbarrier, 1);
cutlass::arch::fence_view_async_shared();
copy(tma_A.with(tma_load_mbarrier), tAgA(_, ik), tAsA);
set_barrier_transaction_bytes(tma_load_mbarrier, tma_transaction_bytes);
wait_barrier(tma_load_mbarrier, ik & 1);
```

通常只让 warp 0 的一个 elected lane 发起 TMA；mbarrier 记录本次事务的字节数，TMA 完成后更新 barrier phase。`ik & 1` 让同一个 barrier 在奇偶 phase 间复用。barrier 初始化必须通过 async-proxy fence 对 TMA 可见，kernel 结束前用 `invalidate_barrier` 释放它。

### 3. TMA 与 MMA 的衔接

```cpp
copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);
gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
```

TMA 只负责大块 gmem→smem；warp 仍需用 `ldmatrix`/通用 copy 把 SMEM fragment 装入寄存器，再执行 warp MMA。不要把“descriptor 搬运”误认为已经完成了寄存器布局转换。

### 4. TMA store 与 epilogue

```cpp
copy(r2s_tiled_copy_d, tDrD_r2s, tDsD_r2s);
tma_store_fence();
copy(tma_D, tDsD, tDgD);
tma_store_arrive();
tma_store_wait<0>();
```

先由线程把 accumulator 按 D 的 SMEM layout 写入共享内存，再由一个线程发起 TMA store。`tma_store_fence` 保证异步代理能看到 SMEM 写入，`arrive/wait` 保证 kernel 返回前全局写完成。

## 小作业

1. 把 `cp.async` 与 TMA 的数据流画成两张图，标注“谁发起搬运、谁等待完成、谁读取 SMEM”。
2. 打印 `tma_transaction_bytes`，分别计算 FP16 A/B 的 `128×64` tile 需要多少字节。
3. 删去 `fence_view_async_shared` 或 `tma_store_fence`（仅在可恢复的实验分支），说明为什么可能出现未完成/不可见数据。
4. 将 `kBlockK` 改为 32，检查 TMA descriptor 的连续维度和对齐限制；记录编译器或 `can_implement` 报错。
5. 比较 `tma_load_store.py` 与 `09-pipelining/pipelining.py` 的运行时间，并解释 TMA 减少了哪些线程级指令与同步开销。

