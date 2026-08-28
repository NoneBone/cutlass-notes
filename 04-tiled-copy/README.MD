# 04 · Tiled Copy：让数据搬运匹配 MMA 的线程布局

本目录在 `03-tiled-mma` 的基础上学习 CuTe 的 `TiledCopy`。上一节使用 `autovec_copy` 完成全局内存（gmem）与寄存器（rmem）之间的复制；这一节显式构造 A、B、C、O 各自的复制布局，让**搬运数据的线程/向量布局与 MMA fragment 的布局对齐**。

示例仍然是一个 block 计算完整的 `32×32×16` tile：

```text
A: [32, 16]，BF16
B: [32, 16]，BF16（数学计算使用 B.T）
C: [32, 32]，FP32 累加器输入
O: [32, 32]，BF16 输出

O = (A @ B.T).to(bf16)                 # MM
O = (A @ B.T + C).to(bf16)             # MMA
```

底层 MMA atom 仍是 SM80 `m16n8k8`，通过 TiledMma 扩展为 256 个线程（8 个 warp）。本节的新增内容不是更换数学运算，而是明确回答：**每个线程从哪里读哪些元素、把它们放到哪个寄存器位置，以及如何按同一布局写回。**

## 文件导览

| 文件 | 作用 | 建议阅读 |
| --- | --- | --- |
| `tiled_copy.py` | 编译 C++ 扩展，构造 `32×32×16` 输入并验证 MM/MMA | 先读 |
| `tiled_copy.cu` | 用 `Copy_Atom`、`TiledCopyA/B/C/O` 和 `retile` 实现复制 | 重点精读 |
| `cutedsl_tiled_copy.py` | CuTe DSL Python 版本，展示 `make_tiled_copy_*` 和 `partition/retile` | 对照 C++ |
| `profile.sh` | Nsight Compute / Nsight Systems 命令示例 | 最后尝试 |

需要 CUDA GPU（SM80 或更高）、PyTorch CUDA 扩展编译环境，以及匹配的 CUTLASS / CuTe DSL。运行方式：

```bash
cd 04-tiled-copy
python tiled_copy.py
python cutedsl_tiled_copy.py
```

首次运行 C++ 版本会即时编译。当前示例仍是固定尺寸、单 block tile，并没有共享内存 staging 或多 tile K 循环；`TiledCopy` 只是把 gmem↔rmem 的访问组织得更明确，为后续 shared-memory copy 做准备。

## `tiled_copy.py`：Python 端值得学习的部分

### 1. 形状和精度路径与上一节保持一致

```python
M, N, K = 32, 32, 16
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
c = torch.randn(M, N, device="cuda", dtype=torch.float32)

kernel_output = lib.tiled_copy_bf16_bf16_bf16_fp32(a, b, c)
torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
```

函数名中的 `bf16_bf16_bf16_fp32` 表示 BF16 A、BF16 B、BF16 输出、FP32 累加器。即使最终输出是 BF16，参考实现也先在 FP32 中完成 `addmm`，最后再 `.bfloat16()`；这样才能和 kernel 的精度流比较。

传入 `None` 时走 MM 路径，C++ 端创建并清零 FP32 累加器；传入 C 时走 MMA 路径，C 会先被复制到累加器。这里的 `c` 在 kernel 中可能被原地更新，若后面还要使用原始 C，应传 `c.clone()`。

### 2. 验证函数如何帮助区分“计算错误”和“舍入差异”

`compare_matrix` 将 kernel 输出和参考输出转换成 FP32，然后报告最大绝对误差、平均绝对误差和 L2 相对误差。由于最终结果是 BF16，少量末位差异是正常的；优先观察相对误差是否低于阈值，而不是要求每个元素逐位相等。

## `cutedsl_tiled_copy.py`：显式 TiledCopy 的核心模式

### 1. 为每个 operand 构造与 MMA 匹配的 copy

```python
universal = cute.nvgpu.CopyUniversalOp()
atom_a = cute.make_copy_atom(universal, mA.element_type)
atom_b = cute.make_copy_atom(universal, mB.element_type)
atom_c = cute.make_copy_atom(universal, mC.element_type)
atom_o = cute.make_copy_atom(universal, mO.element_type)

tiled_copy_a = cute.make_tiled_copy_A(atom_a, tm)
tiled_copy_b = cute.make_tiled_copy_B(atom_b, tm)
tiled_copy_c = cute.make_tiled_copy_C(atom_c, tm)
tiled_copy_o = cute.make_tiled_copy_C(atom_o, tm)
```

`CopyUniversalOp` 是通用复制操作，`make_copy_atom` 将它和元素类型（BF16 或 FP32）组合成 copy atom。`make_tiled_copy_A/B/C` 再依据 `tm`（TiledMma）生成与 A/B/C fragment 相容的线程和值布局。O 的形状和 C 一样，因此复用 C 方向的布局，只把元素类型换成输出类型。

### 2. `partition_S` 与 `retile`：把源张量和目标 fragment 接到同一复制布局

```python
thr_copy_a = tiled_copy_a.get_slice(tid)
tAgA = thr_copy_a.partition_S(gA)
tArA = thr_copy_a.retile(tCrA)
cute.copy(tiled_copy_a, tAgA, tArA)
```

- `get_slice(tid)`：取得当前线程在 TiledCopy 中负责的那一片；
- `partition_S(gA)`：把 gmem tile 按 copy 的 source（S）布局切成线程可访问的视图；
- `retile(tCrA)`：把 MMA 的寄存器 fragment 重新解释成 copy 的 destination（D）布局；
- `cute.copy`：按这套对应布局执行复制。

这不是重新排列数据的额外 kernel，而是创建不同的静态视图。理解 `partition_S → retile → copy`，就掌握了 CuTe 中“让 copy 布局适配既有 fragment”的常用套路。

### 3. C 的预载和 O 的写回使用不同方向

```python
if cutlass.const_expr(is_gemm):
    tCrC.fill(0.0)
else:
    thr_copy_c = tiled_copy_c.get_slice(tid)
    tCgC_g2r = thr_copy_c.partition_S(gC)
    tCrC_g2r = thr_copy_c.retile(tCrC)
    cute.copy(tiled_copy_c, tCgC_g2r, tCrC_g2r)

# MMA 后：寄存器 → 输出
thr_copy_o = tiled_copy_o.get_slice(tid)
tOrC_r2g = thr_copy_o.retile(tCrO)
tOgO_r2g = thr_copy_o.partition_D(gO)
cute.copy(tiled_copy_o, tOrC_r2g, tOgO_r2g)
```

MM 路径无需读取 C，直接清零累加器；MMA 路径才执行 C 的 gmem→rmem 复制。计算完成后，O 使用相反方向的 rmem→gmem 复制：寄存器 fragment 是 source，`gO` 是 destination，因此使用 `retile` 和 `partition_D`。

### 4. 为什么先转换再写 O

```python
tCrO = cute.make_fragment_like(tCrC, out_dtype)
tCrO.store(tCrC.load().to(out_dtype))
```

`tCrC` 仍是 FP32 累加器，`tCrO` 才是 BF16 输出 fragment。转换发生在寄存器中，再由 O-side TiledCopy 写入显存，避免额外的转换 kernel，也避免过早以 BF16 累加。

## `tiled_copy.cu`：C++ 对照阅读重点

### `Copy_Atom` 与四种 `TiledCopy` 类型

```cpp
using Copy_op = AutoVectorizingCopy;
using CopyA_atom = Copy_Atom<Copy_op, ComputeTypeA>;
using CopyB_atom = Copy_Atom<Copy_op, ComputeTypeB>;
using CopyC_atom = Copy_Atom<Copy_op, ComputeTypeC>;
using CopyO_atom = Copy_Atom<Copy_op, OutType>;

using TiledCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
using TiledCopyB = decltype(make_tiled_copy_B(CopyB_atom{}, TiledMMA{}));
using TiledCopyC = decltype(make_tiled_copy_C(CopyC_atom{}, TiledMMA{}));
using TiledCopyO = decltype(make_tiled_copy_C(CopyO_atom{}, TiledMMA{}));
```

这段类型代码说明了 CuTe 的设计：copy 的元素类型和线程布局都是类型的一部分，编译期即可推导每个线程的访问片段。A/B/C/O 分别使用正确的元素类型，O 虽复用 C 的布局，但其元素类型是 `OutType`。

### `retile_S/retile_D` 与 DSL 的对应关系

C++ kernel 中：

```cpp
Tensor tAgA = g2r_thr_copy_a.retile_S(tCgA);
Tensor tArA = g2r_thr_copy_a.retile_D(tCrA);
copy(g2r_tiled_copy_a, tAgA, tArA);
```

DSL 中对应为：

```python
tAgA = thr_copy_a.partition_S(gA)
tArA = thr_copy_a.retile(tCrA)
cute.copy(tiled_copy_a, tAgA, tArA)
```

两者表达的是同一件事：source 视图必须符合复制源布局，destination 视图必须符合复制目标布局。不同版本 API 的命名略有差异，阅读时应先看“谁是 source、谁是 destination”，不要只背函数名。

### 固定尺寸的限制

C++ host 端仍以模板参数固定 `M=32, N=32, K=16`，kernel 内使用 `(0,0,0)` tile 坐标；虽然代码计算了 `grid`，当前输入尺寸实际上只产生一个 block。因此把 Python 的 M/N 改大并不会自动得到通用 GEMM，还需要使用 `blockIdx` 选择 tile、沿 K 循环并处理边界。

## 小作业

1. 运行两个脚本，记录 `Block Size`、MM/MMA 两个 case 的相对误差和输出 dtype。解释为什么输出是 BF16，但 C 仍是 FP32。
2. 在 `cutedsl_tiled_copy.py` 中打印或观察 A 的 `partition_S` 与 `retile` 视图，说明它们为什么元素数量相同但布局不同。
3. 将 `cute.copy(tiled_copy_a, tAgA, tArA)` 临时替换为上一节的 `cute.autovec_copy(tAgA, tArA)`，验证结果是否改变，并解释显式 TiledCopy 主要改变的是访问布局/向量化策略，而不是 GEMM 数学结果。
4. 将 A、B 设为全 1，C 设为全 2。由于 K=16，预测 MM 输出为 16，MMA 的 FP32 累加结果为 18，最终 BF16 输出应接近这两个值。
5. 思考题：如果以后把 gmem→rmem 改为 gmem→smem→rmem，需要新增哪些 tensor view、copy 和同步步骤？先列出数据流，再阅读后续 shared-memory 示例。

完成本节后，可继续阅读 `05-block-mma`，观察更高层次的 block-level MMA 如何组织 tile、线程和内存层次。
