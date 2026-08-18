// Phase 6 occupancy-report CLI. Prints one JSON line (and a human-readable
// line to stderr) per invocation, combining:
//   - a per-SM THEORETICAL occupancy report from the CUDA Runtime's own
//     occupancy API (src/common/occupancy.cuh) for the requested kernel's
//     actual compiled register/shared-memory footprint, and
//   - for kernel families whose grid size depends on problem shape (gemm),
//     a GRID-level utilization figure: how many of this device's SMs the
//     requested shape's grid can even keep busy, independent of per-SM
//     occupancy.
//
// This is Phase 6's stand-in for `ncu`'s occupancy/launch-statistics
// sections, which are unavailable in this environment (ERR_NVGPUCTRPERM;
// see docs/decisions/0014-phase6-profiling-evidence-strategy.md). Every
// number here comes from the CUDA Runtime API, not from a hardware counter.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "common/device_query.hpp"
#include "common/occupancy.cuh"
#include "kernels/gemm/gemm_register_tiled.cuh"
#include "kernels/gemm/gemm_tiled.cuh"
#include "kernels/reduction/reduce_sequential_addressing.cuh"
#include "kernels/reduction/reduce_vectorized_coarsened.cuh"
#include "kernels/reduction/reduce_warp_shuffle.cuh"
#include "kernels/transpose/transpose_naive.cuh"
#include "kernels/transpose/transpose_tiled.cuh"

namespace {

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

// Prints the OccupancyReport plus family/variant/shape context as one JSON
// line to stdout and a human-readable summary to stderr.
//
// `problem_m`/`problem_n` echo the raw --m/--n CLI values this
// grid_blocks_total was computed from (-1 sentinel if not given/not
// applicable to this family -- same convention as grid_blocks_total). This
// is committed so that downstream consumers (analysis/generate_plots.py)
// can join records to a problem shape by VALUE instead of relying on the
// on-disk record order matching some other file's sweep order.
void emit(const std::string& family, const std::string& variant,
          const kernelforge::OccupancyReport& r, long long grid_blocks_total,
          int device_sm_count, long long problem_m, long long problem_n) {
  double grid_utilization = 0.0;
  int theoretical_max_resident_blocks = 0;
  if (grid_blocks_total > 0 && r.max_active_blocks_per_sm > 0) {
    theoretical_max_resident_blocks = device_sm_count * r.max_active_blocks_per_sm;
    const double frac = static_cast<double>(grid_blocks_total) /
                         static_cast<double>(theoretical_max_resident_blocks);
    grid_utilization = frac < 1.0 ? frac : 1.0;
  }

  std::printf(
      "{\"record_type\":\"occupancy_report\",\"kernel_family\":\"%s\",\"variant\":\"%s\","
      "\"block_size\":%d,\"registers_per_thread\":%d,\"static_shared_bytes_per_block\":%zu,"
      "\"dynamic_shared_bytes_per_block\":%zu,\"max_threads_per_block_this_kernel\":%d,"
      "\"device_sm_count\":%d,\"device_max_threads_per_sm\":%d,"
      "\"max_active_blocks_per_sm\":%d,\"achieved_threads_per_sm\":%d,"
      "\"occupancy_fraction\":%.6f,\"suggested_block_size\":%d,"
      "\"suggested_min_grid_size\":%d,\"problem_m\":%lld,\"problem_n\":%lld,"
      "\"grid_blocks_total\":%lld,"
      "\"theoretical_max_resident_blocks\":%d,\"grid_utilization_fraction\":%.6f}\n",
      json_escape(family).c_str(), json_escape(variant).c_str(), r.block_size,
      r.registers_per_thread, r.static_shared_bytes_per_block, r.dynamic_shared_bytes_per_block,
      r.max_threads_per_block_this_kernel, r.device_sm_count, r.device_max_threads_per_sm,
      r.max_active_blocks_per_sm, r.achieved_threads_per_sm, r.occupancy_fraction,
      r.suggested_block_size, r.suggested_min_grid_size, problem_m, problem_n, grid_blocks_total,
      theoretical_max_resident_blocks, grid_utilization);

  std::fprintf(stderr,
               "occupancy family=%s variant=%s block_size=%d regs/thread=%d "
               "static_smem=%zuB dyn_smem=%zuB  max_active_blocks/SM=%d  "
               "achieved_threads/SM=%d/%d (%.1f%% of device max)",
               family.c_str(), variant.c_str(), r.block_size, r.registers_per_thread,
               r.static_shared_bytes_per_block, r.dynamic_shared_bytes_per_block,
               r.max_active_blocks_per_sm, r.achieved_threads_per_sm, r.device_max_threads_per_sm,
               r.occupancy_fraction * 100.0);
  if (grid_blocks_total > 0) {
    std::fprintf(stderr, "  grid_blocks=%lld/%d SMs*blocks-per-SM (%.1f%% grid utilization)",
                 grid_blocks_total, theoretical_max_resident_blocks, grid_utilization * 100.0);
  }
  std::fprintf(stderr, "\n");
}

} // namespace

int main(int argc, char** argv) {
  std::string family;
  std::string variant;
  int block_size = 256;
  long long m = -1, n = -1; // gemm-only, for grid-level utilization

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* flag) -> const char* {
      if (a == flag && i + 1 < argc) return argv[++i];
      return nullptr;
    };
    if (const char* v = next("--family")) family = v;
    else if (const char* v = next("--variant")) variant = v;
    else if (const char* v = next("--block-size")) block_size = std::atoi(v);
    else if (const char* v = next("--m")) m = std::atoll(v);
    else if (const char* v = next("--n")) n = std::atoll(v);
  }

  if (family.empty() || variant.empty()) {
    std::fprintf(stderr,
                  "occupancy_report: usage: --family {transpose|reduction|gemm} --variant <name> "
                  "[--block-size N] [--m M --n N (gemm grid utilization)]\n");
    return 2;
  }
  if (block_size <= 0) {
    // A bad or unparsable --block-size (std::atoi returns 0 for garbage
    // input) must not reach cudaOccupancyMaxActiveBlocksPerMultiprocessor --
    // fail loudly here, before touching the GPU, rather than let a
    // zero-valued-but-schema-valid occupancy_report line get committed.
    std::fprintf(stderr, "occupancy_report: --block-size must be > 0; got %d\n", block_size);
    return 2;
  }

  try {
    const auto device_info = kernelforge::query_device_info(0);
    const int sm_count = device_info.multiprocessor_count;

    if (family == "transpose") {
      kernelforge::OccupancyReport r;
      if (variant == "v1_naive") {
        r = kernelforge::kernels::transpose_naive_query_occupancy();
      } else if (variant == "v2_tiled") {
        r = kernelforge::kernels::transpose_tiled_query_occupancy();
      } else {
        std::fprintf(stderr, "occupancy_report: unknown transpose variant '%s'\n",
                     variant.c_str());
        return 2;
      }
      // Grid-level utilization for a square n x n transpose, if --n given
      // (must match transpose_{naive,tiled}.cu's own grid formula: 32x32
      // tiles, ceil(n/32) in each dimension).
      long long grid_blocks = -1;
      if (n > 0) {
        const long long tiles = (n + 31) / 32;
        grid_blocks = tiles * tiles;
      }
      // transpose is n x n square -- only --n is meaningful (m stays the
      // -1 not-applicable sentinel).
      emit(family, variant, r, grid_blocks, sm_count, /*problem_m=*/-1, /*problem_n=*/n);
    } else if (family == "reduction") {
      kernelforge::OccupancyReport r;
      if (variant == "v2_sequential_addressing") {
        r = kernelforge::kernels::reduce_sequential_addressing_query_occupancy(block_size);
      } else if (variant == "v3_warp_shuffle") {
        r = kernelforge::kernels::reduce_warp_shuffle_query_occupancy(block_size);
      } else if (variant == "v4_vectorized_coarsened") {
        r = kernelforge::kernels::reduce_vectorized_coarsened_query_occupancy(block_size);
      } else {
        std::fprintf(stderr, "occupancy_report: unknown reduction variant '%s'\n",
                     variant.c_str());
        return 2;
      }
      // reduction's grid size doesn't depend on a --m/--n problem shape in
      // this CLI (both stay the -1 not-applicable sentinel).
      emit(family, variant, r, -1, sm_count, /*problem_m=*/-1, /*problem_n=*/-1);
    } else if (family == "gemm") {
      kernelforge::OccupancyReport r;
      int tile = 0;
      if (variant == "v3_tiled") {
        r = kernelforge::kernels::gemm_tiled_query_occupancy();
        tile = 32; // kernels/gemm/gemm_tiled.cuh: kGemmTileDim
      } else if (variant == "v4_register_tiled") {
        r = kernelforge::kernels::gemm_register_tiled_query_occupancy();
        tile = 64; // kernels/gemm/gemm_register_tiled.cuh: kGemmRegBM/kGemmRegBN
      } else {
        std::fprintf(stderr, "occupancy_report: unknown gemm variant '%s'\n", variant.c_str());
        return 2;
      }
      long long grid_blocks = -1;
      if (m > 0 && n > 0) {
        const long long block_rows = (m + tile - 1) / tile;
        const long long block_cols = (n + tile - 1) / tile;
        grid_blocks = block_rows * block_cols;
      }
      emit(family, variant, r, grid_blocks, sm_count, /*problem_m=*/m, /*problem_n=*/n);
    } else {
      std::fprintf(stderr, "occupancy_report: unknown --family '%s' (expected transpose, "
                            "reduction, or gemm)\n",
                   family.c_str());
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "occupancy_report: FATAL: %s\n", e.what());
    return 1;
  }
}
