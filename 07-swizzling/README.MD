# 07 · Swizzling：用共享内存布局减少 bank conflict

本目录继续使用 `06-block-copy` 的数据流，但把 A、B、C、O 的共享内存布局改成带 swizzle 的 `ComposedLayout`，并在 16-bit 操作数的 SMEM→寄存器路径使用 `ldmatrix`。目标是让 warp 协作读取矩阵时尽量分散到不同 shared-memory bank，减少 bank conflict。

```text
gmem --(128-bit cp.async)--> swizzled smem
     --(ldmatrix / copy)--> registers --(MMA)--> registers
     --(R2S)--> swizzled smem --(S2G)--> gmem
```

问题尺寸仍是 `M=N=128, K=64`，一个 block 使用 256 个线程。示例测试三种 dtype 组合：

- FP16 输入、FP32 累加、FP16 输出（验证）；
- FP16 输入、FP16 累加、FP16 输出（仅练习，不与 FP32 参考比较）；
- BF16 输入、FP32 累加、BF16 输出（验证）。

## 文件导览

| 文件 | 作用 | 建议阅读 |
| --- | --- | --- |
| `swizzling.py` | 编译 C++ 扩展，运行三种精度配置并验证 | 先读 |
| `swizzling.cu` | 定义 `Swizzle` SMEM 布局、cp.async、ldmatrix/stmatrix copy | 重点精读 |
| `cutedsl_swizzling.py` | CuTe DSL 版本，展示 `ComposedLayout` 与专用 copy op | 对照 C++ |
| `profile.sh` | Nsight Compute 分析命令 | 最后尝试 |

运行方式：

```bash
cd 07-swizzling
python swizzling.py
python cutedsl_swizzling.py
```

需要 SM80 或更高的 CUDA GPU。当前是单 block、无流水线的教学版本，SMEM 布局是二维的，没有额外 stage 维；真实大 GEMM 还需要结合多 stage pipeline 和边界处理。

## `swizzling.py`：Python 端值得学习的部分

### 1. 同一 kernel 验证不同输入/累加精度

```python
# FP16 输入、FP32 累加
kernel_output = lib.swizzling_fp16_fp16_fp16_fp32(a, b, c.clone())
torch_output = torch.addmm(c, a.float(), b.T.float()).half()

# BF16 输入、FP32 累加
kernel_output = lib.swizzling_bf16_bf16_bf16_fp32(a, b, c)
torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
```

比较函数名时，先确认“输入类型、累加器类型、输出类型”三者，不要只看最后一个 dtype。FP16 输入的 PyTorch `matmul` 通常已使用 FP32 累加，所以参考代码显式转换输入并在最后 `.half()`；BF16 case 同理在最后 `.bfloat16()`。

### 2. FP16 累加配置为何只运行不验证

```python
_ = lib.swizzling_fp16_fp16_fp16_fp16(a, b, c.clone().half())
```

这个配置让 C 和累加器都为 FP16。它可以用来观察 kernel 是否能编译、运行和分析，但 FP16 累加的舍入顺序与 PyTorch 默认 FP32 累加不同，不能直接拿 FP32 参考值判断逐元素一致。先理解“可运行”与“数值验证基准匹配”是两件事。

### 3. `ENABLE_PROF` 与确定性随机数

脚本用 `torch.cuda.manual_seed_all(9527)` 让重复运行时输入可复现，并在 profiling 模式下跳过比较输出。调试布局或比较不同 copy 策略时，固定随机输入便于区分代码变化和输入变化造成的差异。

## `cutedsl_swizzling.py`：Swizzle 与专用内存指令

### 1. 构造 `ComposedLayout`

```python
swz = cute.make_swizzle(3, 3, 3)
atom_AB = cute.make_composed_layout(
    swz, 0,
    cute.make_layout((8, inner_AB), stride=(inner_AB, 1)),
)
sA_layout = cute.tile_to_shape(atom_AB, (M, K), order=(0, 1))
sB_layout = cute.tile_to_shape(atom_AB, (N, K), order=(0, 1))
```

先定义一个小的 layout atom，再用 `make_composed_layout` 把 swizzle 叠加到地址映射上，最后 `tile_to_shape` 扩展成完整 SMEM tile。逻辑形状仍是 `(M,K)` 或 `(N,K)`，改变的是元素映射到 bank 的方式，而不是矩阵数学顺序。

`Swizzle<3,3,3>` 可以粗略理解为对部分地址位做 XOR 重排；三个参数控制参与重排的位宽和位置。实际位级映射应以 CuTe 的 layout 打印结果为准，不建议凭直觉把它当作简单转置。

### 2. G2S、S2R 和 R2S 使用不同 copy atom

```python
g2s_op = cute.nvgpu.cpasync.CopyG2SOp(...)
ldm_op_ab = cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4)
stm_op = cute.nvgpu.warp.StMatrix8x8x16bOp(False, 2)
```

- G2S 使用 128-bit `cp.async`，适合合并的全局→共享内存搬运；
- 16-bit A/B 的 S2R 使用 `ldmatrix`，直接按 warp 的矩阵布局从 SMEM 装入寄存器；
- SM90 且输出为 16-bit 时，R2S 可使用 `stmatrix`；其他情况回退到通用 store。

同一份数据在不同内存层次需要不同布局和指令，这是本节相比 `06-block-copy` 的主要进步。

### 3. FP32 C 预载时的中间 fragment

```python
tCrC_pre = cute.make_fragment_like(tCrC, mC.element_type)
cute.copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r)
tCrC.store(tCrC_pre.load().to(tCrC.element_type))
```

当 C 是 FP16 而累加器是 FP32 时，SMEM→寄存器的 copy 两端位宽必须匹配，因此先复制到与 C 同类型的 `tCrC_pre`，再通过 `.to()` 转成 FP32 的 `tCrC`。这是一个很实用的调试线索：遇到 CuTe copy 位宽不匹配时，应拆成“同类型搬运 + 寄存器转换”。

### 4. 通过 swizzled SMEM 完成 epilogue

```python
tCrO = cute.make_fragment_like(tCrC, out_dtype)
tCrO.store(tCrC.load().to(out_dtype))
# R2S: tCrO -> sO；同步后 S2G: sO -> gO
```

输出也先写入带 swizzle 的 `sO`，然后使用沿 N 方向连续的 TV layout 写回 gmem。R2S layout 服务于 MMA fragment，S2G layout 服务于全局连续写入；中间的 SMEM 缓冲区把两种需求隔离开。

## `swizzling.cu`：建议精读的 C++ 片段

### 1. `Swizzle` 布局和 SMEM 大小

```cpp
using SmemLayoutAtomAB = decltype(composition(
    Swizzle<3, 3, 3>{},
    make_layout(make_shape(Int<8>{}, Int<min(64, kBlockK)>{}), ...)));
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomAB{}, make_shape(Int<kBlockM>{}, Int<kBlockK>{}), Step<_1, _2>{}));
```

A/B 共享同一个 swizzled atom，再铺成完整 tile。`cosize(SmemLayout)` 计算的是布局覆盖的地址范围，乘以元素大小后得到动态 SMEM 申请量；带 swizzle 的 layout 可能存在“逻辑元素数”和地址跨度不同的情况，所以不要只用 `M×K×sizeof(T)` 猜测 SMEM 大小。

### 2. `sizeof` 选择 ldmatrix/stmatrix

```cpp
using Copy_S2R_op_A = std::conditional_t<sizeof(ComputeTypeA) == 2,
                                         SM75_U32x4_LDSM_N,
                                         AutoVectorizingCopy>;
```

`ldmatrix` 面向 16-bit 矩阵数据；FP32 C 不能直接使用相同的 ldmatrix atom，因此回退到通用复制。输出 R2S 也根据架构和元素位宽选择 `SM90_U32x2_STSM_N` 或通用复制。精度类型不仅影响数值，还会改变可选的硬件搬运指令。

### 3. 为什么同步仍不可省略

G2S 的 `cp_async` 完成后要 `cp_async_fence`、`cp_async_wait<0>` 和 `__syncthreads()`，保证 S2R 读取的是完整 SMEM tile。MMA 完成后，R2S 写入 `sO`，再次 `__syncthreads()` 后才能由 S2G 读取。Swizzle 只改变地址布局，不能替代异步复制等待或 block 同步。

## 小作业

1. 运行脚本，记录三种 dtype 配置是否成功；解释为什么 FP16 累加配置只运行不做常规比较。
2. 画出 `gmem → smem → rmem → MMA → rmem → smem → gmem` 数据流，并在每条箭头标注使用的 copy 类型。
3. 在 DSL 中把 `cute.make_swizzle(3, 3, 3)` 临时替换为无 swizzle 的普通 layout，比较结果是否改变，并说明数学结果不变但 bank conflict/性能可能改变。
4. 计算一次 128-bit `cp.async` 对 FP16、BF16、FP32 分别搬运多少元素，并解释为什么 C 的复制布局沿 N 方向组织。
5. 思考题：如果把 K 改为 128 且引入两个 pipeline stage，SMEM layout 需要增加哪些维度？`cp_async_wait_group`、`__syncthreads()` 和下一 stage 的读写顺序应如何安排？

完成本节后，可继续阅读 `08-dynamic-mma`，观察运行时 dtype/shape 参数如何与静态 layout 结合。
