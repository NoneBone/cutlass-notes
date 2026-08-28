# 08 · Dynamic MMA：动态尺寸与边界谓词

本目录把前一节的单 block GEMM 扩展为真正的动态矩阵乘：运行时读取 `M、N、K`，用二维 block grid 覆盖任意大小的输出，并循环处理 `ceil_div(K, BLK_K)` 个 K tile。默认 tile 为 `128×128×64`、每个 block 256 线程；A/B 仍通过 128-bit `cp.async` 搬运，SMEM 使用 swizzle。

```text
gmem A/B --(predicated cp.async)--> swizzled smem
        --(S2R / ldmatrix)--> registers --(MMA over K tiles)--> accumulator
        --(R2S + predicate)--> gmem C/O
```

## 文件导览与运行

| 文件 | 作用 | 建议 |
| --- | --- | --- |
| `dynamic_mma.py` | 即时编译 `dynamic_mma.cu`，遍历多组尺寸并和 PyTorch 比较 | 先读 |
| `dynamic_mma.cu` | 动态 grid、K 循环、M/N/K 边界 predicate 的 CUDA/CuTe 实现 | 精读 |
| `cutedsl_dynamic_mma.py` | 同一算法的 CuTe DSL 写法 | 对照阅读 |
| `profile.sh` | Nsight Compute 性能分析命令 | 最后尝试 |

```bash
cd 08-dynamic-mma
python dynamic_mma.py
python cutedsl_dynamic_mma.py
```

N、K 需要是 8 的倍数，以满足示例中的向量化 copy 对齐要求；M 可以不是 tile 的整数倍。完整测试尺寸较多，初学时可先把 `Ms/Ns/Ks` 改成 `[130]`、`[136]`、`[70]`，专门观察边界块。

## `dynamic_mma.py`：值得学习的代码段

### 1. 用同一入口覆盖 GEMM 与 MMA

```python
kernel_output = lib.dynamic_mma_fp16_fp16_fp16_fp32(a, b, None)       # C 不提供：纯 GEMM
kernel_output = lib.dynamic_mma_fp16_fp16_fp16_fp32(a, b, c.clone())   # 提供 C：C + A @ B.T
```

`None` 让 C++ 端分配零初始化的 C；传入 `c` 则走加法累加路径。参考结果要匹配累加精度：FP16/BF16 输入通常先转 FP32，再在最后转回输出类型。

### 2. 尺寸笛卡尔积与误差检查

```python
exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]
...
re = relative_error(kernel_output, torch_output)
is_correct = re < 0.001
```

这种测试方式能同时覆盖整 tile、部分 M/N tile 和多个 K tile。`max_diff`、`mean_diff` 与相对误差一起看，比只打印一个元素更适合判断数值 kernel。

## `dynamic_mma.cu`：建议精读的片段

### 1. block grid 与局部 tile

```cpp
int bidx = blockIdx.x, bidy = blockIdx.y;
auto coord = make_coord(bidy, bidx, _);
Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
```

行方向由 `bidy` 覆盖 M，列方向由 `bidx` 覆盖 N；A/B 的第三个逻辑模式保留 K tile 迭代。host 端以 `ceil_div(N, 128) × ceil_div(M, 128)` 发射 grid。

### 2. `domain_offset` 与 K 残块

```cpp
auto k_residue = K - size<1>(gA) * size<2>(gA);
gA = domain_offset(make_coord(0, k_residue, 0), gA);
gB = domain_offset(make_coord(0, k_residue, 0), gB);
```

最后一个 K tile 可能不足 64 个元素。把第一个 tile 的逻辑起点平移到 residue 后，后续完整 tile 可以复用同一布局；第 0 tile 额外检查 K 坐标，避免越界读。

### 3. identity tensor 生成 M/N/K 谓词

```cpp
Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
tApA_g2s(m, 0) = get<0>(tAcA_g2s(0, m, 0)) < m_max_coord;
...
copy_if(g2s_tiled_copy_a, tApA_g2s, src, dst);
```

identity tensor 保存每个 tile 元素的逻辑坐标，而不是数据值。将它按相同 tiled copy 分区后，可为每个线程生成 M/N 边界 mask；`copy_if` 对无效元素不发起全局读。无效的 SMEM 槽位先 `clear` 为 0，保证 MMA 不会使用未初始化数据。

### 4. 编译期布尔分支与输出转换

```cpp
constexpr bool IsCvtPrecision = needs_precision_conversion<ComputeTypeC, OutType>();
BOOL_SWITCH(is_gemm, IsGemm, [&] { dynamic_mma<Spec, IsGemm, IsCvtPrecision><<<...>>>(); });
```

把运行时是否提供 C 转成模板布尔值，让编译器删除无关分支。若累加器和输出类型不同，先把 fragment 转成 `OutType`，再经 SMEM 写回；否则直接写回 C。

## 小作业

1. 将尺寸改为 `M=130,N=136,K=70`，画出 grid 中哪些 block 是边界块，并解释三个 predicate 分别保护什么。
2. 暂时注释 `clear(tAsA_g2s)` 或把 `copy_if` 改成 `copy`，观察越界输入下的结果，说明为何必须清零和屏蔽。
3. 在 Python 中增加 BF16 的 `M=257,N=129,K=192` 测试，并记录相对误差。
4. 修改 `kBlockK` 为 32，检查 `k_residue`、向量化 copy 对齐和 SMEM 大小的变化。
5. 思考：为何动态的 M/N/K 仍要求 tile 形状、MMA 指令和 copy atom 在编译期固定？

