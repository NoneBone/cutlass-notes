# 03 · Tiled MMA：把一个 MMA atom 扩展成更大的 tile

本目录在前两节“单个 warp、单个 MMA 指令”的基础上，学习 CuTe 的 `TiledMma`：把一个 SM80 `m16n8k8` MMA atom 沿 M、N、K 方向复制/扩展，让多个 warp 共同计算更大的矩阵 tile。

本例使用 BF16 输入、FP32 累加，最终输出 BF16：

```text
A: [M, K] = [32, 16]，BF16
B: [N, K] = [32, 16]，BF16（数学表达式中使用 B.T）
C: [M, N] = [32, 32]，FP32 初始累加值
O: [M, N] = [32, 32]，BF16 输出

O = (A @ B.T)                 # MM：累加器清零
O = (A @ B.T + C_in).to(bf16) # MMA：FP32 累加后再转换
```

基础指令的 tile 是 `(16, 8, 8)`。本例把它扩展为 `(32, 32, 16)`：M 方向复制 2 份，N 方向复制 4 份，K 方向通过值级扩展复制 2 份。最终一个 block 使用 `2 × 4 × 1 × 32 = 256` 个线程，即 8 个 warp。

## 文件导览

| 文件 | 作用 | 建议 |
| --- | --- | --- |
| `tiled_mma.py` | 编译 `tiled_mma.cu`，生成输入并验证 MM / MMA 两种结果 | 先读，理解调用和形状 |
| `tiled_mma.cu` | 定义 `KernelSpec`、`TiledMMA` 和 CUDA kernel 的线程/值级扩展 | 配合 Python 对照 |
| `cutedsl_tiled_mma.py` | 用 CuTe DSL Python 写出同一个 `(32, 32, 16)` TiledMma | 再读，理解 DSL 写法 |
| `profile.sh` | Nsight Compute 分析命令 | 最后尝试 |

需要 CUDA GPU（SM80 或更高）、PyTorch CUDA 扩展编译环境，以及与仓库匹配的 CUTLASS / CuTe 头文件或 `nvidia-cutlass-dsl` 包。

```bash
cd 03-tiled-mma
python tiled_mma.py
python cutedsl_tiled_mma.py
```

首次运行 C++ 版本会即时编译。脚本会把 kernel 输出与 PyTorch 参考值比较，打印最大误差、平均误差和相对误差；BF16 输出时，参考实现要保持“FP32 中累加，最后转 BF16”的顺序。

## `tiled_mma.py`：Python 端如何准备和验证更大的 tile

### 1. 输入形状从 `16×8×8` 变为 `32×32×16`

```python
Ms = [32]
Ns = [32]
Ks = [16]

a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
c = torch.randn(M, N, device="cuda", dtype=torch.float32)
```

这里的 `M/N/K` 与 `01-minimal-gemm` 不同：它们描述整个 TiledMma block 要覆盖的 tile，而不是单条 MMA 指令的形状。`b` 仍按 `[N, K]` 存储，所以参考计算使用 `b.T`。

示例仍然把尺寸写在 Python 列表中并只测试一个组合；这有利于学习布局，但不是动态、通用的 GEMM 接口。对应 C++ 绑定把 `32, 32, 16` 作为模板参数，并且会检查输入形状。

### 2. MM 与 MMA 的参考实现

```python
kernel_output = lib.tiled_mma_bf16_bf16_bf16_fp32(a, b, None)
torch_output = torch.matmul(a.float(), b.T.float()).bfloat16()

kernel_output = lib.tiled_mma_bf16_bf16_bf16_fp32(a, b, c.clone())
torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
```

传入 `None` 时，C++ kernel 将 FP32 累加器清零；传入 `c.clone()` 时，先把 FP32 C 载入累加器，再执行乘加。两条参考路径都先提升 A、B 到 FP32 做数学运算，最后才转换成 BF16，才能与 kernel 的精度流匹配。

`clone()` 用来保护原始 C，避免 kernel 的原地累加改变参考输入。`compare_matrix` 会转成 FP32 后计算误差；BF16 的舍入差异不要求逐元素完全相等。

## `cutedsl_tiled_mma.py`：TiledMma 的三个关键参数

### 1. 指令 atom、线程级复制和值级复制

```python
MMA_INST_MNK = (16, 8, 8)
ATOM_LAYOUT_MNK = (2, 4, 1)
VAL_EXPAND_MNK = (1, 1, 2)

PERMUTATION_MNK = (
    ATOM_LAYOUT_MNK[0] * VAL_EXPAND_MNK[0] * MMA_INST_MNK[0],
    ATOM_LAYOUT_MNK[1] * VAL_EXPAND_MNK[1] * MMA_INST_MNK[1],
    ATOM_LAYOUT_MNK[2] * VAL_EXPAND_MNK[2] * MMA_INST_MNK[2],
)
assert PERMUTATION_MNK == (M, N, K)
```

可以把三组参数分别理解为：

- `MMA_INST_MNK`：硬件单条 MMA 指令处理的基本形状 `(16, 8, 8)`；
- `ATOM_LAYOUT_MNK`：把 atom 分配给多少个线程组/warp，`(2, 4, 1)` 表示 M 方向 2 份、N 方向 4 份；
- `VAL_EXPAND_MNK`：每个线程的值片段沿哪个维度扩展，本例沿 K 扩展 2 倍。

因此输出 tile 为 `M=2×1×16=32`、`N=4×1×8=32`，归约维为 `K=1×2×8=16`。这里的 K 扩展不是增加线程数，而是让同一线程布局拥有两组 K 方向的 operand 值，并由 `cute.gemm` 完成两次 K 子块的乘加。

### 2. 构造 `TiledMma` 并由线程数决定 block 大小

```python
op = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.BFloat16, cutlass.Float32, MMA_INST_MNK
)
tm = cute.make_tiled_mma(
    op,
    atom_layout_mnk=ATOM_LAYOUT_MNK,
    permutation_mnk=PERMUTATION_MNK,
)
num_threads = tm.size  # 256
```

`MmaF16BF16Op` 选择 BF16 输入、FP32 累加的 SM80 warp MMA。`make_tiled_mma` 把 atom 按线程布局扩展；`tm.size` 是这套布局所需的线程总数，不能随意写成 32。随后 host 端以 `block=(num_threads, 1, 1)` 启动一个含 8 个 warp 的 block。

C++ 版本使用等价的 `MMAThrLayout` 和 `MMATileLayout` 类型参数构造 `TiledMMA`；学习时可把两种写法并排对照：DSL 用元组表达布局，C++ 用 CuTe 的静态 `Layout` / `Tile` 类型表达布局。

### 3. kernel 内的统一数据流

```python
gA = cute.local_tile(mA, tiler=(M, K), coord=(0, 0))
gB = cute.local_tile(mB, tiler=(N, K), coord=(0, 0))
gC = cute.local_tile(mC, tiler=(M, N), coord=(0, 0))

thr_mma = tiled_mma.get_slice(tid)
tCgA = thr_mma.partition_A(gA)
tCrA = tiled_mma.make_fragment_A(tCgA)
# ...
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

与最小示例相同，先用 `local_tile` 选出 block tile，再用 `get_slice(tid)` 得到当前线程在整个 TiledMma 中的分片。区别在于：这里每个线程的分片属于 256 线程的整体布局，而不是单个 warp 的布局。`cute.gemm` 会按照 TiledMma 的 atom、线程布局和值布局，发出覆盖完整 `(32, 32, 16)` tile 所需的 MMA 操作。

### 4. FP32 累加结果在寄存器中转换为 BF16

```python
tCrO = cute.make_fragment_like(tCrC, out_dtype)
tCrO.store(tCrC.load().to(out_dtype))
tCgO = thr_mma.partition_C(gO)
cute.autovec_copy(tCrO, tCgO)
```

`tCrC` 是 FP32 累加器 fragment，`tCrO` 是 BF16 输出 fragment。转换发生在寄存器中，之后直接写到 `mO`；不会先把 FP32 结果写回显存、再额外启动转换 kernel。这也解释了为什么 host 端同时准备 FP32 的 `c` 和 BF16 的 `o`。

## C++ `tiled_mma.cu` 中值得对照的片段

### `KernelSpec` 如何把布局参数变成类型

```cpp
static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});
using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));
```

这段代码是 TiledMma 的“尺寸公式”：先从 MMA atom 的 `Shape_MNK` 取出基本形状，再乘上线程级和值级扩展，最后把结果用于静态 tile 和线程布局。静态信息越丰富，CuTe 越能在编译期生成正确的 fragment 映射，但也意味着改变 tile 形状通常需要重新编译。

### 为什么 `grid` 目前仍只有一个有效 tile

```cpp
dim3 grid((N + Spec::kTileN - 1) / Spec::kTileN,
          (M + Spec::kTileM - 1) / Spec::kTileM);
```

这行看起来像是在支持多 block，但本例绑定的 `M=32, N=32` 与 tile 一样大，所以 `grid=(1,1,1)`。kernel 内仍固定从 `(0,0,0)` 取 tile，且 host 端检查输入必须正好是 `[32,16]`、`[32,16]`、`[32,32]`。因此它展示的是“一个 block 内的 TiledMma”，还不是任意尺寸的完整 GEMM。要做通用版本，还需要按 blockIdx 计算 tile 坐标、沿 K 循环，并处理边界。

## 小作业

1. 运行两个脚本，记录 block size、两种 case 的输出 dtype 和 `RE`。解释为什么 block size 是 256 而不是 32。
2. 手算 `MMA_INST_MNK=(16,8,8)`、`ATOM_LAYOUT_MNK=(2,4,1)`、`VAL_EXPAND_MNK=(1,1,2)` 产生的 `(M,N,K)`，并在代码中打印 `PERMUTATION_MNK` 验证。
3. 将 `VAL_EXPAND_MNK` 临时改为 `(1,1,1)`，观察断言或编译结果。说明为什么 K=8 已不足以覆盖问题的 K=16。
4. 将输入改为全 1、C 改为全 2。因为 K=16，预测 MM 输出为 16，MMA 的 FP32 累加结果为 18，最终 BF16 输出也应分别接近 16 和 18。
5. 思考题：若想支持 `M=N=64, K=16`，需要如何修改 tile 坐标、grid 和边界处理？为什么仅把 Python 列表中的 `M/N` 改成 64 不够？

完成本节后，可以继续阅读 `04-tiled-copy`，学习如何把 TiledMma 前后的全局内存访问组织成更高效的 TiledCopy。
