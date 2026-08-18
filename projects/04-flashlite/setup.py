"""Build script for the flashlite package's CUDA extension.

ADR 0005 (docs/decisions/0005-cuda-extension-integration-route.md): the
integration route is pybind11 + torch.utils.cpp_extension.CUDAExtension
(D6's recommended default), not a hand-rolled CMake+libtorch build. Package
metadata (name, version, dependencies) lives in pyproject.toml; this file
exists only because ext_modules with a custom BuildExtension cmdclass is
not yet expressible in pyproject.toml alone for torch's build helper.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Relative to this file's directory (setup.py's own directory), NOT
# absolute: setuptools' egg_info/sdist manifest step rejects absolute
# source paths ("setup() arguments must always be /-separated paths
# relative to the setup.py directory").
CUDA_NAIVE_DIR = os.path.join("src", "flashlite", "cuda_naive")
CUDA_TILED_DIR = os.path.join("src", "flashlite", "cuda_tiled")
CUDA_ONLINE_SOFTMAX_DIR = os.path.join("src", "flashlite", "cuda_online_softmax")
CUDA_FUSED_DIR = os.path.join("src", "flashlite", "cuda_fused")

# Pinned to sm_89 (RTX 4090), matching KernelForge's ADR 0001
# (../02-kernelforge/docs/decisions/0001-target-gpu-architecture.md) -- the
# only GPU this repo is built and tested against. Overridable via the
# standard TORCH_CUDA_ARCH_LIST env var if the caller sets it explicitly
# before invoking pip/setup.py.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

# Static project metadata (name, version, description, dependencies,
# packages) all live in pyproject.toml's [project] / [tool.setuptools...]
# tables -- setup.py intentionally passes ONLY what pyproject.toml cannot
# yet express (ext_modules with a custom BuildExtension cmdclass), to avoid
# declaring the same field in two places.
setup(
    ext_modules=[
        CUDAExtension(
            name="flashlite._cuda_naive",
            sources=[
                os.path.join(CUDA_NAIVE_DIR, "bindings.cpp"),
                os.path.join(CUDA_NAIVE_DIR, "attention_naive.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "-lineinfo"],
            },
        ),
        # V2 (Phase 3): shared-memory-tiled QK^T/PV, same build route (ADR
        # 0005), a separate extension module so V1 and V2 stay independently
        # importable/buildable -- matching the "variants live side-by-side"
        # convention (spec Part 2) rather than one module with an internal
        # variant switch.
        CUDAExtension(
            name="flashlite._cuda_tiled",
            sources=[
                os.path.join(CUDA_TILED_DIR, "bindings.cpp"),
                os.path.join(CUDA_TILED_DIR, "attention_tiled.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "-lineinfo"],
            },
        ),
        # V3 (Phase 4): online-softmax row normalization (docs/online-softmax.md),
        # kernel 1/3 unchanged from V2. Separate extension module, same
        # "variants live side-by-side" convention as V1/V2 above.
        CUDAExtension(
            name="flashlite._cuda_online_softmax",
            sources=[
                os.path.join(CUDA_ONLINE_SOFTMAX_DIR, "bindings.cpp"),
                os.path.join(CUDA_ONLINE_SOFTMAX_DIR, "attention_online_softmax.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "-lineinfo"],
            },
        ),
        # V4 (Phase 5): single fused kernel, no [B,H,S,S] scores buffer at
        # all (attention_fused.cuh's header; ADR 0011).
        CUDAExtension(
            name="flashlite._cuda_fused",
            sources=[
                os.path.join(CUDA_FUSED_DIR, "bindings.cpp"),
                os.path.join(CUDA_FUSED_DIR, "attention_fused.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
