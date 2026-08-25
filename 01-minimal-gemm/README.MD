# 01 · Minimal GEMM：从一条 Tensor Core MMA 指令开始

本目录是 CUTLASS / CuTe 的最小可运行 GEMM 示例。它不追求通用尺寸和高吞吐，而是把问题缩小到一条 SM80 Tensor Core 指令：

```text
A: [M, K] = [16, 8]
B: [N, K] = [ 8, 8]       # 数学表达式中使用 B.T
C: [M, N] = [16, 8]

C = A @ B.T               # MM 路径
C = A @ B.T + C_in        # MMA（预载 C）路径
```

`mma.m16n8k8` 表示一次 warp 级矩阵乘累加会处理 `16 × 8 × 8` 的 tile。整个问题刚好只有一个 tile，所以只启动一个 warp（32 个线程），没有分块循环、共享内存或流水线；这使你可以专注理解 CuTe 的张量布局、线程分片和寄存器 fragment。

## 文件导览

| 文件 | 角色 | 建议阅读顺序 |
| --- | --- | --- |
| `minimal_gemm.py` | PyTorch 驱动：即时编译 `minimal_gemm.cu`、构造数据并验证结果 | 先读 |
| `minimal_gemm.cu` | C++ CuTe 内核：全局内存 → 寄存器 → MMA → 全局内存 | 配合第一个 Python 文件读 |
| `cutedsl_minimal_gemm.py` | 同一内核的 CuTe DSL（Python）版本 | 再读 |
| `profile.sh` | 使用 Nsight Compute 分析内核的命令示例 | 最后尝试 |

## 运行前提与方式

需要 CUDA GPU，且计算能力至少为 SM80（例如 A100、RTX 30 系列及更新架构）。C++ 版本需要 PyTorch CUDA 扩展编译环境和本仓库的 `third-party/cutlass` 子模块；CuTe DSL 版本还需要与当前 CUDA / PyTorch 匹配的 `nvidia-cutlass-dsl` 包。

```bash
cd 01-minimal-gemm

# C++ CuTe 内核：首次运行会即时编译，因而较慢
python minimal_gemm.py

# CuTe DSL Python 内核：会先编译两个 constexpr 特化版本
python cutedsl_minimal_gemm.py
```

两者都会输出 `Max diff`、`Mean diff` 和相对误差（`RE`）。此示例以 `RE < 0.001` 为正确阈值。FP16 的舍入和累加顺序不同，末位出现轻微差异是正常的，不必期待与 PyTorch 逐元素完全相同。

## `minimal_gemm.py`：Python 如何驱动 C++ CuTe 内核

### 1. 用 `torch.utils.cpp_extension.load` 即时构建扩展

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
sources = [os.path.join(current_dir, "minimal_gemm.cu")]
os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(
    map(str, torch.cuda.get_device_capability())
)
lib = load(name="minimal_gemm", sources=sources, ...)
```

这段代码连接了 Python 与 CUDA/C++：`load` 调用编译器，将 `minimal_gemm.cu` 编译成可从 Python 调用的扩展，随后可用 `lib.minimal_gemm(...)` 调用由 pybind 暴露的函数。

重点看 `TORCH_CUDA_ARCH_LIST`：它根据当前 GPU 的计算能力生成目标架构，避免默认编译到不匹配的架构。`extra_cuda_cflags` 中的 `-I.../cutlass/include` 让编译器找到 CUTLASS / CuTe 头文件；`--generate-line-info` 则有助于 Nsight Compute 将指令映射回源码。

### 2. 明确矩阵形状与 `B.T`

```python
a = torch.randn(M, K, device="cuda", dtype=torch.half)
b = torch.randn(N, K, device="cuda", dtype=torch.half)
c = torch.randn(M, N, device="cuda", dtype=torch.half)

kernel_output = lib.minimal_gemm(a, b, None)
torch_output = torch.matmul(a, b.T)
```

初学者最容易混淆的是 `b` 的存储形状。示例让 `b` 按 `[N, K]` 连续存储，所以数学上的 GEMM 是 `A[M, K] @ B.T[K, N]`，输出才是 `[M, N]`。这对应 CUDA 文件中 `mB` 的 `(N, K)` stride 描述。传入 `None` 表示不提供初值 C，C++ 内核会把累加器清零，得到纯矩阵乘。

### 3. 用合适的参考实现验证 FP16 结果

```python
kernel_output = lib.minimal_gemm(a, b, c.clone())
torch_output = torch.addmm(c, a, b.T)
compare_matrix(kernel_output, torch_output)
```

第二个调用传入 `c.clone()`，内核会先读取 C 到累加器，再执行 MMA，因此是 `A @ B.T + C`。`clone()` 很关键：它避免把后面要作为参考答案的原始 `c` 原地改掉。

`compare_matrix` 会先转成 `float`，再计算最大绝对误差、平均绝对误差和 L2 相对误差。累加情形使用 `torch.addmm`，因为它更接近 GEMM 的融合累加语义；不同实现的 FP16 累加顺序仍可能带来微小差异。

## `cutedsl_minimal_gemm.py`：用 Python 表达同一套 CuTe 概念

### 1. 一个 warp 的 tile、线程切片和寄存器 fragment

```python
gA = cute.local_tile(mA, tiler=(M, K), coord=(0, 0))
gB = cute.local_tile(mB, tiler=(N, K), coord=(0, 0))
gC = cute.local_tile(mC, tiler=(M, N), coord=(0, 0))

thr_mma = tiled_mma.get_slice(tid)
tCgA = thr_mma.partition_A(gA)
tCrA = tiled_mma.make_fragment_A(tCgA)
```

`local_tile` 从全局张量取出当前 block 要处理的 tile；本例问题与 tile 一样大，因此坐标始终是 `(0, 0)`。`get_slice(tid)` 按线程号取得该线程在 MMA 布局中的职责。`partition_A/B/C` 产生“该线程该从全局内存访问的元素”，`make_fragment_A/B/C` 创建形状相匹配的寄存器容器。

这是理解 CuTe 最值得反复看的模式：**布局决定每个线程看到哪些元素，fragment 决定这些元素如何留在寄存器中供 Tensor Core 使用。** 你无需手算 32 个线程各自的索引，`TiledMma` 会根据 MMA 指令布局完成映射。

### 2. 从内存搬入寄存器，执行 MMA，再写回

```python
cute.autovec_copy(tCgA, tCrA)
cute.autovec_copy(tCgB, tCrB)

if cutlass.const_expr(is_gemm):
    tCrC.fill(0.0)
else:
    cute.autovec_copy(tCgC, tCrC)

cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
cute.autovec_copy(tCrC, tCgC)
```

这正是最小 GEMM 的完整数据流：全局内存（`tCg*`）→ 寄存器（`tCr*`）→ Tensor Core `gemm` → 全局内存。`tCrC` 同时是输入累加器与输出：清零时计算纯 MM，预载 C 时计算 `C += A @ B.T`。`autovec_copy` 让 CuTe 根据布局选择可向量化的复制方式，而不用手写每线程 load/store。

### 3. 指令选择、`constexpr` 特化与编译

```python
op = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.Float16, cutlass.Float16, (M, N, K)
)
tiled_mma = cute.make_tiled_mma(op)

gemm_clear = cute.compile(minimal_gemm, ..., True, ...)
gemm_accum = cute.compile(minimal_gemm, ..., False, ...)
```

`MmaF16BF16Op` 选择 SM80 的 warp 级 `m16n8k8` 指令，输入和累加器均为 FP16；`tiled_mma.size` 因而为 32，launch 的 block 恰好是一个 warp。`is_gemm` 声明为 `cutlass.Constexpr[bool]`，编译时会删除未选分支，所以 `True` 和 `False` 必须各编译一个版本，运行时调用时不再传这个参数。

再看 `make_cute_tensor`：它通过 DLPack 零拷贝地把 PyTorch tensor 交给 CuTe DSL，并用 `mark_layout_dynamic(leading_dim=1)` 标记连续的最后一维。这是 Python 框架张量接入 CuTe JIT 的常见入口。

## 小作业

1. 将 `a` 和 `b` 临时改为全 1，`c` 改为全 2（保持原有形状和 `torch.half`）。运行前写下预测：纯 MM 的每个元素应为多少？累加 C 的每个元素又应为多少？运行后核对答案。1* 1 *8 +2 = 10.
2. 在任一个 Python 脚本中添加第三个测试：令 `a` 为全 1、`b` 为全 2、`c` 为全 3，并复用 `compare_matrix` 验证结果。请写出你预期的两个输出值。19
3. 思考题：把 Python 里的 `M` 改成 32 会发生什么？结合 `minimal_gemm.cu` 中固定的 `M=16, N=8, K=8` 和只处理 `(0, 0)` tile 的代码，说明它为何不是通用 GEMM。因为 kernel 实现中，M 值写死为 16，导致断言检测不通过，故不是 通用GEMM；如果要实现 GEMM，应当支持 动态输入的 MNK。但也不一定要求 MNK 全部“运行时动态”。高性能 CUTLASS 经常为特定 tile / 数据类型做编译期特化；关键是外层调度、分块和边界处理能让它覆盖更多实际矩阵尺寸。
4. 本例要求 SM80 即可, ncu 观察，说说你的发现。TODO: 

完成后，下一节可阅读 `02-mixed-precision-gemm`，观察输入类型、累加类型和数值验证如何随混合精度而变化。
