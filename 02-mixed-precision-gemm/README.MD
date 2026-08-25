# 02 · Mixed-Precision GEMM：低精度输入，高精度累加

本目录在 `01-minimal-gemm` 的单 warp `16×8×8` MMA 示例上，加入了**混合精度**这一实际 GEMM 中非常重要的设计：A、B 使用较低精度存储和计算，而累加器 C 使用更高精度，以在吞吐、显存带宽和数值精度之间取得平衡。

矩阵布局仍是：

```text
A: [M, K]
B: [N, K]       # 数学表达式使用 B.T
C / O: [M, N]

C = A @ B.T              # MM：累加器清零
C = A @ B.T + C_in       # MMA：预载 FP32 的 C
```

## 本目录涵盖的精度组合

| 配置 | A、B 输入 | 累加器 / 初始 C | 最终输出 | 指令 tile | 条件 |
| --- | --- | --- | --- | --- | --- |
| BF16 → FP32 | BF16 × BF16 | FP32 | FP32 | `m16n8k8` | SM80+ |
| BF16 → BF16 | BF16 × BF16 | FP32 | BF16 | `m16n8k8` | SM80+ |
| FP8 → FP32 | E4M3 × E5M2 | FP32 | FP32 | `m16n8k32` | SM89+、CUDA 12.4+ |
| FP8 → BF16 | E4M3 × E5M2 | FP32 | BF16 | `m16n8k32` | SM89+、CUDA 12.4+ |

一个容易忽略的事实是：第二、四行的“输出是 BF16”不等于“整个计算都用 BF16 累加”。它们仍先用 FP32 累加，再在写回前将最终结果转换为 BF16。这正是混合精度 GEMM 的核心。

## 文件导览与运行

| 文件 | 作用 |
| --- | --- |
| `mixed_precision_gemm.py` | 编译并驱动 C++ CuTe 内核，验证四种精度配置 |
| `mixed_precision_gemm.cu` | C++ CuTe 内核、类型特化和混合 FP8 的内联 PTX MMA 定义 |
| `cutedsl_mixed_precision_gemm.py` | CuTe DSL Python 版本；实现两种 BF16 输入配置 |
| `profile.sh` | 用 Nsight Compute 分析内核的命令示例 |

```bash
cd 02-mixed-precision-gemm

# C++ CuTe 版本：首次运行会即时编译
python mixed_precision_gemm.py

# CuTe DSL 版本：编译并运行两个 BF16 配置
python cutedsl_mixed_precision_gemm.py
```

两种版本均需要 CUDA GPU。BF16 部分要求 SM80 或更高架构；原始 C++ 版本的 FP8 部分会自行检测 GPU 计算能力至少为 8.9 且 CUDA 至少为 12.4，不满足时自动跳过。CuTe DSL 版本特意跳过混合 `E4M3 × E5M2` 的 FP8 路径：当前公开的 warp-level DSL API 不能表达 A、B 使用两种不同 FP8 类型的 MMA。

## `mixed_precision_gemm.py`：把“计算精度”和“输出精度”分开看

### 1. FP32 输出：BF16 输入，但用 FP32 作为参考与结果

```python
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
c = torch.randn(M, N, device="cuda", dtype=torch.float32)

kernel_output = lib.mixed_precision_gemm_fp32_bf16_bf16_fp32(a, b, c.clone())
torch_output = torch.addmm(c, a.float(), b.T.float())
```

函数名可按 `输出_输入A_输入B_累加器` 阅读：`fp32_bf16_bf16_fp32` 表示输出 FP32、两个输入 BF16、累加器 FP32。参考实现把 `a`、`b` 先转为 FP32，这是关键：若直接对 BF16 tensor 调用 `torch.matmul`，参考结果自身可能已经以 BF16 输出，从而无法准确验证“FP32 累加”内核。

`c` 保持 FP32，`torch.addmm(c, ...)` 对应内核的“先载入 FP32 C，再做 MMA”路径。传 `None` 则走纯 MM 路径，并将累加器清零。

### 2. BF16 输出：最后一步才缩窄精度

```python
kernel_output = lib.mixed_precision_gemm_bf16_bf16_bf16_fp32(a, b, c.clone())
torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
```

这里最值得学习的是转换时机：参考答案先完成 FP32 `addmm`，**最后**才调用 `.bfloat16()`。如果一开始就把 C 或中间结果降为 BF16，得到的将是另一种数值计算，而不是本内核执行的计算。

该顺序也解释了为什么第二个配置的 `c` 仍是 `torch.float32`，即使最终返回的 `kernel_output` 是 BF16。

### 3. FP8 路径的能力检查与量化输入

```python
if sm_version >= (8, 9) and cuda_version >= (12, 4):
    a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    a = a_fp32.to(torch.float8_e4m3fn)
    b = b_fp32.to(torch.float8_e5m2)
```

FP8 类型与所需的 MMA/PTX 支持较新，因而脚本先检查硬件和 CUDA 工具链；这是写带新特性的 CUDA 示例时值得复用的保护方式。输入先随机生成 FP32，再显式量化为 E4M3、E5M2。验证时又转回 FP32 做参考：它验证的是“量化后的 A、B”参与计算的结果，而不是原始 FP32 随机数的结果。

## `mixed_precision_gemm.cu`：类型特化如何决定指令与输出路径

### 1. 用 `KernelSpec` 分离四种类型角色

```cpp
template <typename OutType_, typename ComputeTypeA_, typename ComputeTypeB_,
          typename ComputeTypeC_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8>
struct KernelSpec {
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;
  // 根据 A、B、C 的类型选择对应 MMA_op
};
```

不要把 `OutType` 与 `ComputeTypeC` 混为一谈：前者是落到全局内存的输出类型，后者是 Tensor Core MMA 的累加器类型。这个类型分层让同一份内核主体可以覆盖“FP32 累加 → FP32 输出”和“FP32 累加 → BF16 输出”。`MMA_op` 的条件类型选择还展示了：输入/累加器类型的组合会决定实际可用的 Tensor Core 指令。包括 16 位和 8 位两种 MMA 指令。

### 2. 在寄存器中完成 FP32 → BF16 转换

```cpp
if constexpr (!IsCvtPrecision) {
  copy(copy_atom, tCrC, tCgC);
} else {
  auto tCrO = make_tensor_like<OutType>(tCrC);
  copy(tCrC, tCrO);              // FP32 累加器转换为输出类型
  Tensor tCgO = thr_mma.partition_C(gO);
  copy(copy_atom, tCrO, tCgO);
}
```

`tCrC` 是寄存器中的累加器 fragment，始终按 `ComputeTypeC`（本例 FP32）保存。仅当 `OutType` 不同，`IsCvtPrecision` 才在**编译期**为真，内核创建 `tCrO` 并在寄存器中转换，然后写入单独的输出张量 `gO`。

这避免了把 FP32 累加结果先写回显存、再启动额外 kernel 做转换。`if constexpr` 也确保 FP32 输出版本不承担转换分支的运行时开销。

### 3. FP8 混合输入为何需要自定义 MMA 原子

```cpp
asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 ...");
```

代码定义了 `SM90_16x8x32_F32E4M3E5M2F32_TN` 及其 `MMA_Traits`，把混合 FP8 指令包装成 CuTe 可组合的 MMA atom。重点不是立刻记住寄存器布局，而是理解 `MMA_Traits` 的职责：它告诉 CuTe 结果、A、B、累加器的数据类型，M/N/K 指令形状，以及 32 个线程如何分配 operand fragment。等掌握 BF16 路径后，再回头研究这段内联 PTX 会更合适。

## `cutedsl_mixed_precision_gemm.py`：同一思想的 Python DSL 表达

### 1. BF16 输入、FP32 累加器的 MMA 操作

```python
op = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.BFloat16,
    cutlass.Float32,
    (M, N, K),
)
tiled_mma = cute.make_tiled_mma(op)
```

第一个类型参数描述 A/B 的 BF16 输入，第二个描述 FP32 累加器。`tiled_mma` 随后决定每个线程所拥有的 A、B、C fragment；在这个 `16×8×8` 示例中，一个 atom 正好是一个 32 线程 warp。

### 2. 用独立输出缓冲区表达“窄化后写回”

```python
if cutlass.const_expr(is_cvt_precision):
    tCrO = cute.make_fragment_like(tCrC, out_dtype)
    tCrO.store(tCrC.load().to(out_dtype))
    tCgO = thr_mma.partition_C(gO)
    cute.autovec_copy(tCrO, tCgO)
else:
    cute.autovec_copy(tCrC, tCgC)
```

`mC` 始终是 FP32 的累加器输入/输出缓冲区，`mO` 则保存 BF16 输出。`tCrC.load().to(out_dtype)` 明确指出转换发生在寄存器 fragment 中。`out_dtype` 和 `is_cvt_precision` 都是 `Constexpr`，所以 `compile_for` 会为 FP32 输出和 BF16 输出分别生成专用内核。

### 3. 参考实现必须匹配“先累加、后转换”

```python
torch_output = torch.addmm(c, a.float(), b.T.float())
if is_cvt:
    torch_output = torch_output.to(out_torch_dtype)
```

这里将参考值的精度流程写得很清楚：输入提升到 FP32，FP32 中执行矩阵乘和累加，最后才在需要时转为输出类型。调试混合精度 kernel 时，先让参考实现的精度路径与 kernel 一致，比单纯调大误差阈值更可靠。

## 小作业

1. 运行 `mixed_precision_gemm.py`。记录 BF16→FP32 与 BF16→BF16 两种输出的 `Max diff` 和 `RE`；若 FP8 case 被跳过，记录终端显示的 GPU 计算能力与 CUDA 版本。8 种 case 都是 0err。
2. 将 BF16 两组 case 中的 `a`、`b` 临时改为全 1、`c` 改为全 2（保持原形状和 dtype）。预测纯 MM 的每个元素与带 C 累加的每个元素；`K=8`。
3. 在 BF16→BF16 case 中，故意把参考实现改成 `torch.addmm(c.bfloat16(), a, b.T)`，与原代码比较输出。解释它为何不再是正确的 FP32 累加参考。导致了 C 过早丢失精度，正确的参考为：BF16 A、B → FP32 累加器 + FP32 C → 最终转换为 BF16 输出
4. 解释指令的含义 "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "。m16n8k32 不表示“某一个线程”计算 16×8 个元素；是一个 warp 的 32 个线程分配并保存各自的 A/B/C/D 寄存器 fragment，然后共同发射这条 Tensor Core 指令。

| 部分           | 含义                                                         |
| -------------- | ------------------------------------------------------------ |
| `mma`          | Matrix Multiply-Accumulate，矩阵乘加。                       |
| `sync`         | warp 内 32 个线程协作执行这条指令。它不是通用的 block 同步屏障。 |
| `aligned`      | warp 中所有活跃线程必须执行同一条 MMA 指令；这里不是指内存地址对齐。 |
| `m16n8k32`     | 输出 tile 是 `16×8`，归约维度 K 是 `32`。                    |
| `row.col`      | 逻辑上 A 按行主序解释，B 按列主序解释。                      |
| 第一个 `f32`   | 结果 `D` 的类型是 FP32。                                     |
| `e4m3`         | A 的 FP8 类型：4 位指数、3 位尾数。                          |
| `e5m2`         | B 的 FP8 类型：5 位指数、2 位尾数。                          |
| 最后一个 `f32` | 输入累加器 `C` 的类型是 FP32。                               |

5. 上述 MMA 指令， warp 操作字节数计算：
- 必要输入：
  - mnk: D[16, 8] = A[16, 32] × B[32, 8] + C[16, 8] 
  - f32.e4m3.e5m2.f32 的字节数分别为 4.1.1.4
- 合计寄存器 operand payload：1792 B / warp
  - 逻辑寄存器读取：A + B + C = 512 + 256 + 512 = 1280 B  （注：这里算的是 MMA，如果特化为 MM，那么没有 C 的 512）
  - 逻辑寄存器写回：D = 512 B 
- MMA 的计算量: 输出元素个数 * 乘加次数 = m * n * (k * 2) = 8192 FLOPs / warp
- MMA 的计算强度：Arithmetic Intensity = FLOPs / 全局内存传输字节数 = 8192 / 1792 ≈ 4.57 FLOPs/B。

6. 三代架构 TC 指令的区别
```sh
SM80 mma.sync：     1 warp 共同执行
SM90 wgmma：        1 warpgroup（4 warps）共同执行
SM100 tcgen05/UMMA：更大范围的 TC 操作，由选出的线程发射
```

7. blackwell 机器，torch 解释器看到的精度类型
```sh
float
float32
float16
float64

float8_e4m3fn
float8_e4m3fnuz
float8_e5m2
float8_e5m2fnuz
float8_e8m0fnu

float4_e2m1fn_x2
```

建议先真正理解前两种 BF16 配置，再进入 FP8 与内联 PTX 部分；后者的难点主要是硬件指令支持和 fragment 布局，而不是 GEMM 数学本身。
