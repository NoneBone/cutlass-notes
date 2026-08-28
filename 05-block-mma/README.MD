# 05 · Block MMA：让一个线程块计算更大的矩阵块

本目录在 `04-tiled-copy` 的基础上，把一个 block 要处理的区域从 `32×32×16` 扩大到 `128×128×64`，学习 CuTe 如何在 block（线程块）层面组织 MMA。输入仍是 BF16，累加器使用 FP32，最终输出 BF16：

```text
A: [128, 64]，BF16
B: [128, 64]，BF16（数学计算使用 B.T）
C: [128, 128]，FP32 初始累加值
O: [128, 128]，BF16 输出

O = (A @ B.T).to(bf16)
O = (A @ B.T + C).to(bf16)
```

底层 atom 是 SM80 `mma.m16n8k16.bf16.bf16.f32.f32`。它先构成一个 `(32,32,32)` 的 TiledMma tile，再通过 fragment 的 M/N/K 维度扩展覆盖整个 `(128,128,64)` 问题。一个 block 仍使用 256 个线程（8 个 warp），但每个线程持有更大的寄存器 fragment，并由 `cute.gemm` 遍历这些 fragment。

## 文件导览

| 文件 | 作用 | 建议阅读 |
| --- | --- | --- |
| `block_mma.py` | 编译 C++ 扩展、构造 `128×128×64` 数据并验证 MM/MMA | 先读 |
| `block_mma.cu` | 定义 Block-level `TiledMMA`、TiledCopy 和三种 GEMM 主循环写法 | 重点精读 |
| `cutedsl_block_mma.py` | CuTe DSL Python 版本，展示大 fragment 上的 `cute.gemm` | 对照 C++ |
| `profile.sh` | Nsight Compute / Nsight Systems 分析命令 | 最后尝试 |

需要 CUDA GPU（SM80 或更高）、PyTorch CUDA 扩展编译环境，以及匹配的 CUTLASS / CuTe DSL：

```bash
cd 05-block-mma
python block_mma.py
python cutedsl_block_mma.py
```

当前示例是固定尺寸、单 block tile；它展示 block-level MMA 的布局和寄存器组织，还不是任意尺寸的通用 GEMM。由于累加器 fragment 较大，寄存器占用和编译时间也会比前几节明显增加。

## `block_mma.py`：Python 端值得学习的部分

### 1. 形状扩大，但精度流程不变

```python
M, N, K = 128, 128, 64
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
c = torch.randn(M, N, device="cuda", dtype=torch.float32)

kernel_output = lib.block_mma_bf16_bf16_bf16_fp32(a, b, c)
torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
```

和 `03`、`04` 一样，`b` 的存储形状是 `[N,K]`，所以数学表达式要写 `b.T`。函数名中的最后一个 `fp32` 指累加器类型，而不是最终输出类型；此绑定最终返回 BF16。参考值先在 FP32 中计算，最后才转换成 BF16。

### 2. MM / MMA 的输入约定

```python
lib.block_mma_bf16_bf16_bf16_fp32(a, b, None)   # 清零累加器
lib.block_mma_bf16_bf16_bf16_fp32(a, b, c)      # 预载 C 后累加
```

`None` 让 C++ host 创建 FP32 的临时累加器并走 MM 路径；传入 C 则走 MMA 路径，C 会被复制进寄存器并原地更新。若需要保留输入 C，调用时应使用 `c.clone()`。

`compare_matrix` 继续使用最大误差、平均误差和相对误差判断结果。BF16 输出有舍入差异时，应检查精度流程和相对误差，而不是要求逐元素完全相同。

## `cutedsl_block_mma.py`：从 TiledMma 到 Block MMA

### 1. 三个尺寸参数和中间 MMA tile

```python
MMA_INST_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 4, 1)
VAL_EXPAND_MNK = (1, 1, 2)

MMA_TILE_MNK = (32, 32, 32)
```

单条 atom 指令处理 `(16,8,16)`；线程级复制 `(2,4,1)` 和值级 K 扩展 `(1,1,2)` 产生一个 `(32,32,32)` 的 MMA tile。它还不是整个 block 的 `(128,128,64)`，而是每个线程在更大 fragment 中反复使用的基本子 tile。

### 2. 让 fragment 覆盖整个 block tile

```python
tiled_mma = cute.make_tiled_mma(
    op,
    atom_layout_mnk=ATOM_LAYOUT_MNK,
    permutation_mnk=MMA_TILE_MNK,
)

# gA/gB/gC 是 (128,64)/(128,64)/(128,128)
tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(gA))
tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(gB))
tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(gC))
```

`TiledMma` 的 atom 布局决定每条 MMA 的线程分工，`partition_A/B/C` 则把更大的 global tile 映射到每个线程的 fragment。此时 fragment 在逻辑上包含 `MMA_M=4`、`MMA_N=4`、`MMA_K=2` 个子块（`128/32`、`128/32`、`64/32`）。每个线程仍只执行自己那部分寄存器操作，但整个 256 线程 block 合起来覆盖完整矩阵块。

### 3. 一次 `cute.gemm` 遍历 M/N/K fragment

```python
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

与 `03-tiled-mma` 中一个 atom 对应一个小 tile 不同，这里的 `tCrA/tCrB/tCrC` 已经是大 fragment。`cute.gemm` 会依据 TiledMma 的静态布局，在寄存器中遍历 M、N、K 子块并发出所需的多次 MMA 指令；你不必手写 4×4×2 的索引循环。

这是本节最值得掌握的抽象边界：**TiledMma 定义“线程和数据如何对应”，cute.gemm 负责“按这个对应关系完成整个 fragment 的乘加”。**

### 4. 显式 TiledCopy 仍然负责 gmem ↔ rmem

```python
thr_copy_a = tiled_copy_a.get_slice(tid)
tAgA = thr_copy_a.partition_S(gA)
tArA = thr_copy_a.retile(tCrA)
cute.copy(tiled_copy_a, tAgA, tArA)
```

Block MMA 并没有改变内存复制的基本套路：先由 TiledCopy 把整个 block tile 分配给线程，再把数据放入 MMA fragment。区别只是 fragment 的 M/N/K 维度更大，因而每线程需要搬运和保存更多值。

## `block_mma.cu`：建议精读的三段 C++ 代码

### 1. `KernelSpec` 中的尺寸推导

```cpp
static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});
using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));
```

这段把 atom 形状、线程扩展和值扩展合成为静态 TiledMma tile，并据此推导线程数和 fragment 布局。理解这些量之间的乘法关系，是从 tiled MMA 走向 block MMA 的关键。

### 2. 自动主循环与手动主循环

文件中默认启用的是：

```cpp
copy(g2r_tiled_copy_a, tAgA, tArA);
copy(g2r_tiled_copy_b, tBgB, tBrB);
gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
```

下面还保留了两个通过 `#elif` / `#else` 切换的教学版本：一个沿 K 维显式循环，另一个沿 M/N/K 三维子 tile 嵌套循环，并在每次循环中调用 `gemm`。它们帮助你验证 `cute.gemm` 自动遍历的逻辑，但通常会增加源码复杂度。阅读时可比较三种版本访问的 `tCrA/tCrB/tCrC` 子片段。

### 3. 输出转换和固定尺寸边界

```cpp
if constexpr (IsCvtPrecision) {
  auto t = make_tensor_like<OutType>(tCrC);
  copy(tCrC, t);       // FP32 累加器 → BF16 fragment
  copy(copy_atom, t, tCgO);
}
```

FP32 累加器先在寄存器中转换成 BF16，再写回 O。host 端模板参数固定 `M=128,N=128,K=64`，kernel 也从 `(0,0,0)` tile 开始，因此仅修改 Python 输入尺寸不会自动支持更大的矩阵；真正的通用实现还要增加 blockIdx tile 坐标、K 循环和边界掩码。

## 小作业

1. 运行两个脚本，记录 block size、输出 dtype 和 MM/MMA 的相对误差。解释为什么仍是 256 线程，而不是 `128×128` 个线程。
2. 手算 `(16,8,16)`、`(2,4,1)`、`(1,1,2)` 得到的 `(32,32,32)`，再说明 `128×128×64` 相对于它分别扩展了几倍。
3. 在 DSL 代码中把输入 A、B 设为全 1、C 设为全 2。因为 K=64，预测 MM 每个输出为 64，MMA 的 FP32 累加结果为 66，最后 BF16 输出应接近这两个值。
4. 阅读 C++ 中被 `#elif` / `#else` 屏蔽的手动循环，画出它访问的 M/N/K 子块顺序，并与单行 `cute.gemm` 对照。
5. 思考题：若把 block tile 改为 `256×128×64`，哪些静态尺寸、fragment 和线程/寄存器资源需要重新评估？为什么 block tile 不能无限增大？

完成本节后，可继续阅读 `06-block-copy`，学习如何把这些较大的 block-level fragment 通过共享内存和显式复制流水化。
