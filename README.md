# 弯刀笔记

CUTLASS 笔记系列将从最小的 GEMM 实现开始，逐步扩展到包含 CuTe 和各种 CUTLASS 组件，以及 Hopper 和 Blackwell 等新架构的特性，最终实现高性能的融合 GEMM 算子。


## 用法

```bash
git clone https://github.com/ArthurinRUC/cutlass-notes.git

make update  # clone cutlass
```

## 运行示例代码

此 GitHub 仓库中的所有示例代码都可以通过简单地执行 Python 脚本来编译和运行。例如：

```bash
cd 01-minimal-gemm
python minimal_gemm.py
```

## CuTe DSL 版本

每个示例还附带一个 CuTe DSL Python 移植版（`cutedsl_*.py`与原版 `.cu` / `.py` DSL 端口完全跳过了 C++ 构建步骤——无需编译任何东西，只需…… `import` 官方 CuTe DSL Python 包。

这些端口还支持 [TVM-FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html)因此，每个预编译的可调用对象都接受原始数据。 `torch.Tensor` 直接传递参数并运行 `torch.cuda.current_stream()` 无需每次调用​​都进行 dlpack 转换或显式流处理。额外的运行时要求如下：

```bash
pip install \
  "nvidia-cutlass-dsl>=4.3.5" \
  "cuda-python>=12.9" \
  "cuda-bindings>=12.9" \
  "apache-tvm-ffi>=0.1.8" \
  "torch_c_dlpack_ext>=0.1.5"
```

`cuda-bindings` 提供 `cuda.bindings.driver.CUstream` （用于 `@cute.jit` 签名）。 `apache-tvm-ffi` + `torch_c_dlpack_ext` 返回 `--enable-tvm-ffi` 代码生成路径和 `from_dlpack(..., enable_tvm_ffi=True)` 运行时张量绑定。

然后以与运行原始 Python 启动器相同的方式运行 DSL 示例：

```bash
cd 01-minimal-gemm
python cutedsl_minimal_gemm.py
```

DSL端口使用 `@cute.jit` / `@cute.kernel` 表达内核和 `cute.compile(..., options="--enable-tvm-ffi")` 预编译每个 `is_gemm` / 每次运行进行一次数据类型特化。编译模板使用 `make_cute_tensor(...)` （一个 `from_dlpack(..., enable_tvm_ffi=True)` 包装纸）加 `make_fake_stream(use_tvm_ffi_env_stream=True)`运行时，编译后的可调用对象会被直接调用。 `torch.Tensor` 参数 — DSL 同步到 `torch.cuda.current_stream()` 自动（环境流模式）。主机端测试框架镜像原始环境——相同的种子，相同的 `Success / Failed` 总结——通过 CUDA 构建和通过 DSL 构建会打印出结构相同的输出。

## 笔记列表

| 注释 | 摘要 | 链接 |
|---------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **00-简介** | CUTLASS 简介 | [intro](https://zhuanlan.zhihu.com/p/1937220431728845963) |
| **01-minimal-gemm** | - 介绍 CuTe 的基础知识<br>- 从零开始，使用单条 MMA 指令实现了 16x8x8 GEMM 内核。<br>Python内核调用、精度验证和性能基准测试<br>- 使用 Nsight Compute (ncu) 进行性能分析 | [minimal-gemm](https://zhuanlan.zhihu.com/p/1937517614084650073) |
| **02-mixed-precision-gemm** | - 实现混合精度 GEMM，支持不同的输入/输出/累加精度<br>- 探讨内核中数值精度转换的技术细节。<br>- 通过 PTX 指令演示自定义 FP8 GEMM 内核实现（用于 CUTLASS 不支持的 MMA 操作）| [mixed-precision-gemm](https://zhuanlan.zhihu.com/p/1940158874255602181) |
| **03-tiled-mma** | - 介绍 GEMM 算子的关键概念模型：三层平铺。<br>- 详细介绍了 CUTLASS CuTe 中 Tiled MMA 操作的实现。<br>- 解释 Tiled MMA API 中各种参数的用法和语义<br>- 将 GEMM 内核从单指令操作扩展到单块操作 | [tiled-mma](https://zhuanlan.zhihu.com/p/1950555644814946318) |
| **04-tiled-copy** | - 解释 CuTe TiledCopy 的核心原理及其在全局内存和共享内存之间数据移动中的作用<br>- 描述 TiledCopy 的 API 参数和语义<br>- 演示如何在图块级别实现数据复制。<br>- 介绍GPU全局内存访问特性的基础知识 | [tiled-copy](https://zhuanlan.zhihu.com/p/1968745447741972494) |
| **05-block-mma** | - 将 Tiled MMA 扩展到 Block 级别，以进行更大规模的 GEMM 计算<br>- 解释了如何在线程块中组合多个 Tiled MMA 操作<br>- 描述块级别 TiledCopy 和 TiledMMA 的平铺和协调。<br>- 展示了块级 MMA 从全局内存到共享内存再到寄存器的分层数据流 | [block-mma](https://zhuanlan.zhihu.com/p/1970162570636816559) |
| **06-块复制** | - 阶段 A / B / (C) 通过张量核心 MMA 之前的共享内存<br>- 引入 gmem→smem→rmem 数据流，并显式支持 G2S / S2R / R2S / S2G TiledCopies<br>- 使用 128 位 `cp.async` 对于 gmem→smem 路径和 `AutoVectorizingCopy` （`CopyUniversalOp` 对于其余部分，请使用 CuTe DSL。<br>- 详细讲解动态共享内存大小调整以及单个内存块内 A/B/C/O 缓冲区的生命周期 | [block-copy](https://zhuanlan.zhihu.com/p/2004627053077627913) |


## 执照

本项目采用 MIT 许可证授权 - 有关详细信息，请参阅 LICENSE 文件。