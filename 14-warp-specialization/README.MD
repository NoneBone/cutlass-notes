# 14 · Warp Specialization：让不同 warpgroup 分工协作

本目录在 13 的 WGMMA 基础上拆分职责：一个 producer warpgroup 专门发起 TMA（gmem→SMEM），两个 consumer warpgroup 专门执行 WGMMA 和 epilogue。默认一个 CTA 有 3 个 warpgroup、384 个线程；producer 与 consumer 使用不同的寄存器配额和 mbarrier，通过流水线并行工作。

```text
Producer WG (128 threads):  TMA load → full/empty pipeline
Consumer WG0/1 (256 threads): wait → WGMMA → release → epilogue
```

## 文件导览与运行

| 文件 | 作用 |
| --- | --- |
| `warp_specialization.py` / `warp_specialization.cu` | 基础 producer/consumer warp specialization |
| `warp_specialization_pipeline.py` / `warp_specialization_pipeline_api.cu` | 使用 CUTLASS `PipelineTmaAsync` API 的版本 |
| `cutedsl_warp_specialization.py` | CuTe DSL 对照实现 |
| `profile.sh` | Nsight Compute 分析入口 |

```bash
cd 14-warp-specialization
python warp_specialization.py
python warp_specialization_pipeline.py
python cutedsl_warp_specialization.py
```

需要 Hopper/SM90A。先用小尺寸验证正确性，再比较基础手写 barrier 版本与 `PipelineTmaAsync` 封装版本。

## 值得学习的代码段

### 1. 三类 warpgroup 与寄存器重配置

`warp_group_idx` 为 0 时是 producer，1/2 时是 consumer。源码分别调用 `warpgroup_reg_dealloc<kProducerRegisters>()` 和 `warpgroup_reg_alloc<kConsumerRegisters>()`：前者保留较少寄存器给搬运，后者为 WGMMA accumulator 分配更多寄存器。调整这两个常量时应同时观察 spill 与 occupancy。

### 2. Producer 的 TMA pipeline

producer 对每个 stage 调用 `producer_acquire`，确认旧数据已被消费后，通过 `copy(tma_A.with(...), ...)` 和对应的 B copy 发起 TMA，再 `producer_commit`。这把“SMEM 空闲→写入→变满”的状态交给 `PipelineTmaAsync` 管理。

### 3. Consumer 的 WGMMA 主循环

consumer 调用 `consumer_wait(smem_read_pipe)` 等待 stage 填满，随后执行 `warpgroup_fence_operand`、`warpgroup_arrive`、多次 `gemm(tiled_mma, ...)`、`warpgroup_commit_batch` 和 `warpgroup_wait<0>`，最后 `consumer_release` 释放该 stage。两套 acquire/release 方向相反，正是生产者和消费者能够重叠的原因。

### 4. 只同步 MMA 线程

producer 不参与 MMA 的 epilogue，因此基础版本使用 `warpgroup_sync(kNumMmaWarpGroups)`（256 个线程的 named barrier），而不是会等待全部 384 个线程的 `__syncthreads()`。这段设计是 warp specialization 中最容易被忽略、却最关键的同步约束。

### 5. 手写 barrier 与 CUTLASS pipeline API

`warp_specialization.cu` 直接维护 `tma_load_mbarrier`、`mma_consume_mbarrier` 和 phase；`warp_specialization_pipeline_api.cu` 用 `MainloopPipeline` 的 `producer_acquire/commit`、`consumer_wait/release` 封装同样的状态机。建议先读手写版本，再逐一建立 API 对应关系。

### 6. CuTe DSL 的角色分支

`cutedsl_warp_specialization.py` 用 `warp_group_idx`、`PipelineTmaAsync.create` 和 `pipeline.NamedBarrier` 表达同一分工，并将两个 consumer 的 M 子 tile 通过 `warp_group_idx - 1` 切分。DSL 版本适合观察“角色、布局、同步”如何共同决定 kernel 结构。

## 小作业

1. 画出一个 K tile 从 producer 发起 TMA 到 consumer release 的时序图，标出 stage 的 full/empty 状态。
2. 调整 producer/consumer 的寄存器配额，记录寄存器数、spill、occupancy 和运行时间。
3. 将 `warpgroup_sync(kNumMmaWarpGroups)` 临时替换为 `__syncthreads()`，解释同步参与者集合如何改变。
4. 比较基础 CUDA 版本与 `PipelineTmaAsync` 版本，列出至少三组手写 barrier 与 pipeline API 的对应关系。
5. 解释为什么 producer 不执行 WGMMA、consumer 不发起 TMA；如果所有 warpgroup 都做两种工作，会损失哪些并行性？

