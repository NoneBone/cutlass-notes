# 09 · Pipelining：用多 stage 隐藏内存延迟

本目录在 `08-dynamic-mma` 的动态 GEMM 上加入软件流水线。A/B 数据沿 `gmem → smem → registers → Tensor Core` 流动；当当前 tile 做 MMA 时，下一 tile 可以并行进行 `cp.async`，还可以预取下一次 SMEM→寄存器（S2R）数据。默认 `NUM_STAGES=3`，block tile 为 `128×128×64`。

## 文件导览与运行

| 文件 | 作用 |
| --- | --- |
| `pipelining.py` / `pipelining.cu` | C++ CuTe 多 stage 版本，含 G2S 与寄存器预取 |
| `pipelining_no_reg_prefetch.py` / `.cu` | 关闭 S2R 寄存器预取的基线 |
| `cutedsl_pipelining.py` | CuTe DSL 版本，可调 `NUM_STAGES`、`REG_STAGES` |
| `cutedsl_pipelining_profiled.py` | 带 Perfetto/Nsight 打点的进阶版本 |
| `profile.sh` | 性能分析脚本 |

```bash
cd 09-pipelining
python pipelining.py
python pipelining_no_reg_prefetch.py
python cutedsl_pipelining.py
```

可设置 `ENABLE_MMA=1` 测试带 C 累加的路径，`ENABLE_PROF=1` 跳过大量数值比较以便 profiling。

## Python/DSL 重点

### 1. 两种流水线开关

```python
NUM_STAGES = 3       # gmem -> smem 的环形缓冲深度
REG_STAGES = 1       # 1：不预取 S2R；>=2：预取下一个寄存器 fragment
```

`REG_STAGES=1` 对应 `pipelining_no_reg_prefetch.cu`：每个 K tile 先完整加载 A/B，再计算。`REG_STAGES>=2` 把 `k_block+1` 的 S2R 放到当前 MMA 之前，可能隐藏 `ldmatrix/LDS` 延迟，但会增加寄存器占用，甚至 spill；性能不一定更好。

### 2. prologue 与环形 stage 指针

```python
for ik in cutlass.range_constexpr(1, NUM_STAGES - 1):
    ...
    cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(NUM_STAGES - 2)
smem_pipe_read = cutlass.Int32(0)
smem_pipe_write = cutlass.Int32(NUM_STAGES - 1)
```

先填充 `NUM_STAGES-1` 个 stage，再等待足够的异步拷贝完成；`read/write` 指针在 `[0, NUM_STAGES)` 环形回绕。理解这段代码是理解所有 cp.async pipeline 的关键。

## `pipelining.cu`：建议精读的片段

### 1. G2S 与 S2R 重叠

```cpp
cp_async_fence();
cp_async_wait<G2S_Stages - 2>();
...
copy(s2r_tiled_copy_a, tAsA_s2r(_, _, k_next, s2r_smem_pipe), ...);
...
gemm(tiled_mma, tCrC, tCrA(_, _, ik), tCrB(_, _, ik), tCrC);
```

`cp_async_fence/commit` 提交全局到共享内存的异步组，`wait` 保证将要读取的 stage 已到达。寄存器预取把下一 K block 的 A/B 装载提前，从而让 S2R 延迟与 MMA 计算重叠。

### 2. 主循环中的写入、计算与指针回绕

```cpp
if (k_block == 0) {
  copy_if(..., g2s_smem_pipe);  // 发起下一 K tile 的 cp.async
  cp_async_fence();
}
gemm(tiled_mma, tCrC, ...);
```

只在每个 K tile 的第一个 MMA block 发起下一次 G2S，避免重复搬运；stage 写指针递增到 `G2S_Stages` 后回到 0。循环尾部必须 `cp_async_wait<0>()`，确保 epilogue 之前所有数据和写回完成。

### 3. 为什么需要 `__syncthreads()`

异步拷贝的完成通知与 block 内线程对 SMEM 的可见性是两件事。`wait` 后的 `__syncthreads()` 让所有线程在读取同一 stage 前达成屏障；MMA 后写回 SMEM 时也要同步，避免 S2G 读到未完成的结果。

## 性能观察建议

固定 `M=N=K=4096`，分别运行 `REG_STAGES=1` 与 `2`，记录 kernel 时间、寄存器数和 occupancy。更多 stage 能隐藏更长的内存延迟，却会消耗更多 SMEM；寄存器预取则可能造成 spill。`cutedsl_pipelining_profiled.py` 的事件可在 Perfetto 中观察 prologue、compute、wait 的时间关系。

## 小作业

1. 把 `NUM_STAGES` 从 3 改为 2 和 4，确认 SMEM 用量、编译结果和运行时间变化。
2. 对同一尺寸比较 `REG_STAGES=1` 与 `2`，解释“理论重叠更多”却可能变慢的原因。
3. 在主循环中打印/记录 `smem_pipe_read`、`smem_pipe_write` 的序列，画出环形缓冲的读写时间线。
4. 故意把 `cp_async_wait_group` 的等待值改大或删掉一次 `__syncthreads()`，观察错误/不稳定结果，并说明同步职责。
5. 计算一个 `128×128×64` FP16 stage 的 A+B SMEM 字节数，再乘以 `NUM_STAGES`，估算增大 stage 对占用率的影响。

