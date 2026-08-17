cmake_minimum_required(VERSION 3.24)

# --- CUDA compiler default -------------------------------------------------
# Explicit path avoids relying on $PATH resolution (see docs/gpu-model.md /
# scripts/*.sh for why this repo always pins nvcc explicitly). Only applied
# if the caller did not already pass -DCMAKE_CUDA_COMPILER=...
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# --- Target architecture (ADR 0001 / 0002: sm_89, RTX 4090 only) ----------
if(NOT DEFINED KF_CUDA_ARCH)
  set(KF_CUDA_ARCH "89")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${KF_CUDA_ARCH})

project(KernelForge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(KF_BUILD_TESTS "Build KernelForge correctness tests" ON)
option(KF_BUILD_APPS "Build KernelForge device-info/benchmark executables" ON)
option(KF_WARNINGS_AS_ERRORS "Treat host-compiler warnings as errors" OFF)

find_package(CUDAToolkit REQUIRED)

# Default to a Release-ish build unless the caller asked for something else;
# benchmarks measured under an unoptimized build would be meaningless.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

include(cmake/Warnings.cmake)
include(cmake/CudaOptions.cmake)

# Record compiler/flag metadata into a generated header so bench results can
# self-report exactly what compiled them (FR4: "compiler flags").
include(cmake/BuildInfo.cmake)

add_subdirectory(src/common)
add_subdirectory(src/kernels)

if(KF_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(KF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
