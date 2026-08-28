# 10 · GEMM API：用 CUTLASS 高层接口组装 kernel

前几节手写了线程、copy 和 mainloop；本目录改用 CUTLASS 的 `CollectiveMainloop`、`CollectiveEpilogue` 与 `GemmUniversalAdapter` 组装完整 GEMM。你仍需指定 tile、数据类型、布局和架构，但不必自己编写 block 内主循环。默认 tile 为 `128×128×64`、G2S pipeline 为 5 stages。

## 文件导览与运行

| 文件 | 作用 |
| --- | --- |
| `gemm_api.py` | 面向当前 GPU 即时编译 `gemm_api.cu` 并验证多组 M/N/K |
| `gemm_api.cu` | SM80 Tensor Core 的 CUTLASS GEMM API 组装 |
| `gemm_api_sm90a.py` / `.cu` | Hopper SM90A 专用 `CollectiveBuilder` 版本 |
| `profile.sh` | 性能分析入口 |

```bash
cd 10-gemm-api
python gemm_api.py
# Hopper 且需要 SM90A 代码路径时：
python gemm_api_sm90a.py
```

`ENABLE_MMA=1` 才会测试传入 C 的 `C + A @ B.T` 路径；`ENABLE_PROF=1` 可跳过比较输出。N、K 仍需是 8 的倍数以满足示例向量化搬运。

## `gemm_api.cu`：值得学习的代码段

### 1. 用 `KernelSpec` 描述类型、布局和 tile

```cpp
using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<Int<kBlockM_>, Int<kBlockN_>, Int<kBlockK_>>;
using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<G2S_Stages>;
```

`KernelSpec` 把输入/输出/累加类型、Row/Column Major stride、MMA atom、SMEM layout 和 pipeline 深度集中在一个模板中。类型组合不匹配时，`static_assert` 会在编译期给出错误，适合学习“配置即类型”的 CUTLASS 风格。

### 2. Collective mainloop 与 epilogue

```cpp
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<...>;
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementC, ..., LinearCombination<ElementD, 1, ElementAccumulator, ElementCompute, ...>>;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int>,
                                                         CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

Mainloop 负责 cp.async、SMEM→寄存器和 MMA；epilogue 负责 `alpha * accumulator + beta * C`、类型转换和 D 写回。把两者分开后，可以替换调度策略或 epilogue，而不重写整个 kernel。

### 3. Arguments、stride、workspace 与生命周期

```cpp
StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
    {(ElementA*)Aptr, stride_A, (ElementB*)Bptr, stride_B},
    {{1.f, 1.f}, (ElementC*)Cptr, stride_C, (ElementD*)Dptr, stride_D},
    kernel_hw_info};
size_t workspace_size = Gemm::get_workspace_size(arguments);
cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
CUTLASS_CHECK(gemm.can_implement(arguments));
CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
CUTLASS_CHECK(gemm.run(stream));
```

`Arguments` 同时携带问题形状、指针、stride、alpha/beta 和硬件信息。先查询并分配 workspace，再按 `can_implement → initialize → run` 生命周期调用，是 CUTLASS device API 的标准模式；不要把临时 workspace 在 `run` 前释放。

### 4. Python 边界检查与输出类型

```cpp
CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type_a);
CHECK_TORCH_TENSOR_SHAPE(c, M, N);
constexpr bool IsCvtPrecision = needs_precision_conversion<ComputeTypeC, OutType>();
```

扩展入口先验证 dtype/shape，再决定是否额外分配输出张量。比如 FP32 累加、FP16 输出时，D 是独立的 FP16 tensor；同类型时可直接把 C 作为输出缓冲区。理解这一层，有助于排查“kernel 正确但 Python 张量类型不对”的问题。

## SM90A 变体

`gemm_api_sm90a.cu` 使用 `CollectiveBuilder` 根据 `ArchTag=Sm90`、输入类型和布局推导 mainloop/epilogue，并在 Python 中固定 `TORCH_CUDA_ARCH_LIST=9.0a`。它面向 Hopper 的 GMMA/TMA 等调度，配置接口与 SM80 版本相似，但可用的 tile、schedule 和硬件要求不同；应在真实 SM90A GPU 上编译测试，不要把 `9.0a` 二进制当作通用架构。

## 小作业

1. 在 `KernelSpec` 中把 `kBlockM/kBlockN` 改为 64，观察编译期类型和性能变化。
2. 修改 `alpha/beta`（例如 `{2.f, 0.5f}`），用 `torch.addmm` 构造同样参考结果。
3. 传入错误 dtype 或形状，阅读 `CHECK_TORCH_TENSOR_*` 与 `can_implement` 的报错，区分 host 检查和 CUTLASS capability 检查。
4. 将 FP16 输出改为 FP32 输出，追踪 `ElementD`、`IsCvtPrecision` 和 Python 返回 tensor 的变化。
5. 比较手写的 `09-pipelining` 与本目录 API 版本：列出哪些细节被 CUTLASS 隐藏（stage、copy、epilogue），哪些配置仍必须由用户指定。

