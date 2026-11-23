# Tiled Matrix Multiplication (GEMM) and Transformer Kernel Library

Educational kernel implementation with CPU reference validation and optional GPU examples. No training is required or performed. This is a starting implementation, not a reproduction of previously drafted resume performance claims.

## Implemented
- Header-only C++17 blocked FP32 GEMM and RMSNorm.
- Single-head online-softmax attention (causal/noncausal), O(ND) working storage. CPU attention is not a CUDA FlashAttention implementation.
- Optional CUDA shared-memory tiled GEMM with masked boundaries and CUDA-event timing.
- Optional Triton fused RMSNorm with FP32 accumulation and PyTorch reference checks.
- C++ independent-reference tests for irregular shapes, normalization and attention.

## Build on Linux / Windows WSL2
Install a C++17 compiler and CMake. In this directory:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```
Without CMake:
```bash
g++ -O2 -std=c++17 -Iinclude src/check.cpp -o kernel_tests
./kernel_tests
```
## Optional GPU execution
CUDA toolkit + NVIDIA GPU required:
```bash
nvcc -O3 cuda/gemm.cu -o cuda_gemm
./cuda_gemm
```
For Triton use Linux with a compatible PyTorch/CUDA/Triton environment:
```bash
python -m pip install torch triton
python python/triton_rmsnorm.py
```
GPU scripts are unverified on hardware in this delivery environment. CUDA GEMM uses SIMT FP32, not Tensor Cores. Its built-in small constant-input check is a smoke test, not comprehensive GPU validation. Extend with randomized inputs before production use.

## Scope and limitations
Contiguous row-major arrays only; callers own buffer sizes. No full transformer integration, autograd, strided CUDA support, multi-GPU support or claimed peak utilization. The CPU attention implementation is a correctness-oriented online-softmax example. No custom CUDA attention is included. No A100 benchmarks or training runs were performed. See VALIDATION.md for actual checks.

## Benchmark reporting
Record hardware, compiler, clocks, shapes, dtype, warmup, synchronization, repetitions and library baselines. Never copy unmeasured TFLOP/s or bandwidth numbers. Each delivered GPU command produces measurements only when actually run.

CPU benchmark after CMake build: `./build/cpu_benchmark 256 10`. Reports mean wall time and GFLOP/s for blocked GEMM; this is not a vendor BLAS comparison.
