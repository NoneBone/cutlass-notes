# 13 · Warpgroup MMA：用 WGMMA 扩大 Tensor Core 协作范围

本目录从 SM80 的 warp-level MMA 迁移到 Hopper SM90A 的 WGMMA（warpgroup MMA）。一个 warpgroup 包含 128 个线程；示例使用 `64×256×16` 的 WGMMA atom，以 `(2,1,1)` 沿 M 方向铺开，两个 warpgroup（256 线程）共同完成一个 CTA tile。A/B 由 TMA 分阶段搬入共享内存，WGMMA 直接读取适合 GMMA 的 SMEM descriptor。

```text
TMA gmem→SMEM (3 stages) → WGMMA(SMEM A/B → register accumulator)
                          → R2S → TMA store
```

## 文件导览与运行

| 文件 | 作用 |
| --- | --- |
| `warpgroup_mma.py` / `warpgroup_mma.cu` | 基础 WGMMA + TMA GEMM，SM90A |
| `cutedsl_warpgroup_mma.py` | CuTe DSL 对照实现 |
| `warpgroup_mma_multicast.py` / `.cu` | 增加 cluster/TMA multicast 的变体 |
| `profile.sh` | Nsight Compute 分析入口 |

```bash
cd 13-warpgroup-mma
python warpgroup_mma.py
python cutedsl_warpgroup_mma.py
python warpgroup_mma_multicast.py
```

脚本固定 `TORCH_CUDA_ARCH_LIST=9.0a`，需要 Hopper/SM90A；初学时先把尺寸列表缩小，确认 kernel 能编译，再进行 profiling。

## Python 端值得学习的部分

### 1. 接口不变、协作粒度改变

```python
kernel_output = lib.warpgroup_mma_fp16_fp16_fp16_fp32(a, b, None)
torch_output = torch.matmul(a, b.T)
```

Python 端仍是 `(a, b, optional c)`，但 C++ 内部 tile 已改为 `M=128,N=256,K=64`。这体现了“算法接口不变、硬件协作粒度改变”；验证时仍需注意 FP32 accumulator 与最终 FP16/BF16 转换。

### 2. 用两次调用验证 C 累加

```python
o1 = lib.warpgroup_mma_fp16_fp16_fp16_fp16(a, b, None) + c
o2 = lib.warpgroup_mma_fp16_fp16_fp16_fp16(a, b, c)
```

用“先 GEMM 后加 C”和“MMA 直接带 C”比较，可验证 epilogue 与累加路径。`ENABLE_MMA=1`、`ENABLE_PROF=1` 的用法与前几节一致。

## `warpgroup_mma.cu`：建议精读的代码段

### 1. WGMMA atom 与 tiled MMA

```cpp
using MMA_op = SM90_64x128x16_F32F16F16_SS<GmmaMajorA, GmmaMajorB>;
using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));
static constexpr int kWarpgroupNum = kThreadNum / cutlass::NumThreadsPerWarpGroup;
```

`MMA_op` 的 `SS` 表示操作数来自 shared memory；`MMAThrLayout` 决定多个 warpgroup 如何铺开 atom，`MMATileLayout` 决定 CTA 覆盖的 M/N/K。先打印这些 shape，再读主循环会更容易。

### 2. WGMMA 发射协议

```cpp
warpgroup_fence_operand(tCrC);
warpgroup_arrive();
gemm(tiled_mma, tCrC, tCrA(_,_,k_block,smem_read_pipe),
     tCrB(_,_,k_block,smem_read_pipe), tCrC);
warpgroup_commit_batch();
warpgroup_wait<0>();
```

`arrive/commit/wait` 是 WGMMA 的异步发射协议；在下一次使用 accumulator 前必须等待完成。与单 warp `mma.sync` 相比，参与线程数、寄存器 fragment 形态和同步要求都不同。

### 3. 多 stage TMA 与 MMA 消费 barrier

```cpp
initialize_barrier(tma_load_mbarrier[pipe], 1);
initialize_barrier(mma_consume_mbarrier[pipe], kThreadNum);
wait_barrier(tma_load_mbarrier[smem_read_pipe], tma_phase_bit);
arrive_barrier(mma_consume_mbarrier[smem_read_pipe]);
```

一组 barrier 表示“该 SMEM stage 已装满”，另一组表示“MMA 已消费完、producer 可以覆盖”。`smem_read_pipe`、`smem_write_pipe` 和 phase bit 共同实现环形缓冲。

### 4. WGMMA 专用 SMEM layout

```cpp
using SmemLayoutAtomA = decltype(
    cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ...>());
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{}, make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<kStages>{})));
```

`ss_smem_selector` 根据 GMMA major、元素位宽和连续维度选择适合 WGMMA 的 swizzle；逻辑 shape 仍是矩阵坐标，物理地址遵循 GMMA 的 SMEM descriptor 约定。

## multicast 变体

`warpgroup_mma_multicast.cu` 把 12 节的 cluster multicast 思路与 WGMMA 结合：通过 `create_tma_multicast_mask` 复用 A/B tile，并用 cluster launcher 发射。先掌握基础 WGMMA 的两组 warpgroup barrier，再阅读 multicast 版本，避免同时排查两类同步问题。

## 小作业

1. 计算 `64×256×16` WGMMA atom、`(2,1,1)` atom layout 和 256 线程分别覆盖的矩阵范围。
2. 把 `NUM_STAGES=3` 改为 2，画出 producer 写 pipe、consumer 读 pipe 与 phase bit 的变化。
3. 在 `warpgroup_wait<0>()` 前后记录时间，解释为什么不能直接删去 wait。
4. 对比 12 节 warp-level MMA 与本节 WGMMA：列出线程协作粒度、SMEM layout、同步 API 的差异。
5. 阅读 multicast 变体，判断 cluster.x 或 cluster.y 改变时 A/B 哪一个更适合广播，并说明 mask 如何变化。

