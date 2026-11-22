/*
 * Copyright (c) 2025 by SGLang team.
 * Copyright (c) 2024-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SPECULATIVE_SAMPLING_HIP_CUH_
#define SPECULATIVE_SAMPLING_HIP_CUH_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <hipcub/hipcub.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace speculative_hip {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr hipcub::BlockScanAlgorithm SCAN_ALGO = hipcub::BLOCK_SCAN_WARP_SCANS;
constexpr hipcub::BlockReduceAlgorithm REDUCE_ALGO = hipcub::BLOCK_REDUCE_WARP_REDUCTIONS;

// Wavefront size: 32 on RDNA2 (gfx1030), 64 on CDNA (gfx942/gfx950).
// __AMDGCN_WAVEFRONT_SIZE__ is defined by hipcc during device compilation.
// For host-side sizeof(SamplingTempStorage), allocate for worst-case (32 warps).
#ifdef __AMDGCN_WAVEFRONT_SIZE__
static constexpr uint32_t HIP_WARP_SIZE = __AMDGCN_WAVEFRONT_SIZE__;
#else
static constexpr uint32_t HIP_WARP_SIZE = 32;
#endif

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
template <typename T>
__host__ __device__ __forceinline__ constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------
#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      throw std::runtime_error(err_msg.str());                             \
    }                                                                      \
  }

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
  if (deterministic) {                                            \
    constexpr bool DETERMINISTIC = true;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    constexpr bool DETERMINISTIC = false;                         \
    __VA_ARGS__                                                   \
  }

// ---------------------------------------------------------------------------
// Minimal vec_t — array wrapper with load / store / fill / operator[]
// Only float specialisation is needed for this kernel.
// ---------------------------------------------------------------------------
template <typename T, uint32_t N>
struct vec_t {
  T data_[N];

  __device__ __forceinline__ T& operator[](uint32_t i) {
    return data_[i];
  }
  __device__ __forceinline__ const T& operator[](uint32_t i) const {
    return data_[i];
  }

  __device__ __forceinline__ void fill(T val) {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
      data_[i] = val;
  }

  __device__ __forceinline__ void load(const T* ptr) {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
      data_[i] = ptr[i];
  }

  __device__ __forceinline__ void store(T* ptr) const {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
      ptr[i] = data_[i];
  }
};

// ---------------------------------------------------------------------------
// ValueCount
// ---------------------------------------------------------------------------
template <typename T>
struct ValueCount {
  T value;
  int count;

  __device__ ValueCount operator+(const ValueCount& other) const {
    return {value + other.value, count + other.count};
  }
  __device__ ValueCount& operator+=(const ValueCount& other) {
    value += other.value;
    count += other.count;
    return *this;
  }
};

// ---------------------------------------------------------------------------
// BoolDiffOp  (used by BlockAdjacentDifference::SubtractLeft)
// SubtractLeft calls op(current, left_neighbor).  For a monotonically
// non-decreasing boolean CDF array (false…false, true…true), the only
// transition is false→true.  `lhs != rhs` detects this rising edge,
// equivalent to `lhs && !rhs` on monotonic input.  Matches flashinfer.
// ---------------------------------------------------------------------------
struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(const bool& lhs, const bool& rhs) const {
    return lhs != rhs;
  }
};

// ---------------------------------------------------------------------------
// SamplingTempStorage
// Max 32 warps (1024 threads / 32-wide warps). For wave64 targets only 16
// warps exist, but the 32-element array is harmless padding.
// ---------------------------------------------------------------------------
template <
    uint32_t BLOCK_THREADS,
    hipcub::BlockScanAlgorithm SCAN_ALGORITHM,
    hipcub::BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  union {
    float deterministic_scan[BLOCK_THREADS / HIP_WARP_SIZE];
    typename hipcub::BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
    typename hipcub::BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename hipcub::BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename hipcub::BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  } block_prim;

  struct {
    int32_t sampled_id;
    union {
      float value;
      ValueCount<float> pair;
    } block_aggregate;
  };
};

// ---------------------------------------------------------------------------
// DeterministicInclusiveSum — Blelloch-style scan via warp shuffles + smem.
// Ported from flashinfer. On HIP the _sync mask is ignored by hardware but
// we keep the API for source compatibility.
//
// NOTE: This kernel is verified on gfx1030 (RDNA2, waveSize=32).  The
// algorithm is wave-size agnostic (uses runtime `warpSize`), but shared-
// memory sizing uses compile-time HIP_WARP_SIZE.  For CDNA targets (wave64)
// the SamplingTempStorage union layout may need re-validation.
// ---------------------------------------------------------------------------
template <
    uint32_t VEC_SIZE,
    uint32_t BLOCK_THREADS,
    hipcub::BlockScanAlgorithm SCAN_ALGORITHM,
    hipcub::BlockReduceAlgorithm REDUCE_ALGORITHM>
__device__ __forceinline__ void DeterministicInclusiveSum(
    const float* in_data,
    float* out_data,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  const uint32_t ws = warpSize;  // runtime: 32 on RDNA2, 64 on CDNA
  const uint32_t half_ws = ws >> 1;
  float* smem_prefix_sum = temp_storage->block_prim.deterministic_scan;

  // Intra-thread prefix sum
  float thread_data[VEC_SIZE];
  float thread_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    thread_sum += in_data[i];
    thread_data[i] = thread_sum;
  }

  // Up-sweep (reduce) within the warp
  float thread_exclusive_prefix_sum = thread_sum;
  for (uint32_t offset = 1; offset < ws; offset *= 2) {
    float tmp = __shfl_up(thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }

  // Grab the warp total from the last lane
  float warp_sum = __shfl(thread_exclusive_prefix_sum, ws - 1);

  // Clear the last lane for the down-sweep
  if (threadIdx.x % ws == (ws - 1u)) {
    thread_exclusive_prefix_sum = 0;
  }

  // Down-sweep within the warp
  for (uint32_t offset = half_ws; offset >= 1; offset /= 2) {
    float tmp = __shfl_xor(thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
    }
    if ((threadIdx.x + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
  }

  // Write per-warp totals to shared memory
  smem_prefix_sum[threadIdx.x / ws] = warp_sum;
  __syncthreads();

  // First warp scans the per-warp totals
  if (threadIdx.x < ws) {
    const uint32_t num_warps = BLOCK_THREADS / ws;
    float warp_exclusive_prefix_sum = (threadIdx.x < num_warps) ? smem_prefix_sum[threadIdx.x] : 0;

    for (uint32_t offset = 1; offset < ws; offset *= 2) {
      float tmp = __shfl_up(warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }

    if (threadIdx.x % ws == (ws - 1u)) {
      warp_exclusive_prefix_sum = 0;
    }

    for (uint32_t offset = half_ws; offset >= 1; offset /= 2) {
      float tmp = __shfl_xor(warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
      }
      if ((threadIdx.x + 1) % (offset * 2) == offset) {
        warp_exclusive_prefix_sum = tmp;
      }
    }
    if (threadIdx.x < num_warps) {
      smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
    }
  }
  __syncthreads();

  // Combine: warp-prefix + thread-exclusive-prefix + per-element running sum
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_data[i] = smem_prefix_sum[threadIdx.x / ws] + thread_exclusive_prefix_sum + thread_data[i];
  }
}

// ---------------------------------------------------------------------------
// DeviceSamplingFromProb — inverse-CDF sampling from a probability vector.
// ---------------------------------------------------------------------------
template <
    uint32_t VEC_SIZE,
    uint32_t BLOCK_THREADS,
    hipcub::BlockScanAlgorithm SCAN_ALGORITHM,
    hipcub::BlockReduceAlgorithm REDUCE_ALGORITHM,
    bool DETERMINISTIC,
    typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i,
    uint32_t d,
    Predicate pred,
    float u,
    vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  const uint32_t tx = threadIdx.x;

  float prob_greater_than_threshold[VEC_SIZE];
  float inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];

#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
    valid[j] = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
  }

  float aggregate_local = hipcub::BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
                              .template Sum<VEC_SIZE>(prob_greater_than_threshold);
  if (tx == 0) {
    temp_storage->block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
          prob_greater_than_threshold, inclusive_cdf, temp_storage);
    } else {
      hipcub::BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .template InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);
      __syncthreads();
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
    }

    bool greater_than_u_diff[VEC_SIZE];
    // hipcub deduces ITEMS_PER_THREAD from array size
    hipcub::BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .SubtractLeft(greater_than_u, greater_than_u_diff, BoolDiffOp());
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u_diff[j]) {
        atomicMin(&(temp_storage->sampled_id), static_cast<int32_t>((i * BLOCK_THREADS + tx) * VEC_SIZE + j));
      }
    }
    __syncthreads();
  }

  aggregate += aggregate_local;
}

// ---------------------------------------------------------------------------
// TreeSpeculativeSamplingTargetOnly — the __global__ kernel
// ---------------------------------------------------------------------------
template <
    uint32_t BLOCK_THREADS,
    hipcub::BlockScanAlgorithm SCAN_ALGORITHM,
    hipcub::BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    bool DETERMINISTIC,
    typename DType,
    typename IdType,
    typename IdType2>
__global__ void TreeSpeculativeSamplingTargetOnly(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    DType* draft_probs,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

  DType prob_acc = 0.0;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  DType coin = uniform_samples[bx * num_draft_tokens];
  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      DType target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;

      if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
        prob_acc = 0.;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        predicts[last_accepted_retrive_idx] = draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        draft_probs[cur_prob_offset + draft_token_id] = target_probs[cur_prob_offset + draft_token_id];
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;

  // Different coin for the final sampling
  coin = uniform_samples_for_final_sampling[bx];

  // Sample from relu(target_probs - draft_probs)
  DType sum_relu_q_minus_p(0);
  vec_t<DType, VEC_SIZE> q_vec, p_vec;
  DType relu_q_minus_p[VEC_SIZE];
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], DType(0));
    }
    sum_relu_q_minus_p += hipcub::BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                              .template Sum<VEC_SIZE>(relu_q_minus_p);
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = sum_relu_q_minus_p;
  }
  temp_storage.sampled_id = d - 1;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  DType u = coin * sum_relu_q_minus_p;

  DType aggregate_relu_q_minus_p(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }

    vec_t<DType, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], DType(0));
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
        i, d, [&](DType x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, &temp_storage);
    if (aggregate_relu_q_minus_p > u) {
      break;
    }
  }
  __syncthreads();
  predicts[last_accepted_retrive_idx] = temp_storage.sampled_id;
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------
template <typename DType, typename IdType, typename IdType2>
hipError_t TreeSpeculativeSamplingTargetOnly(
    IdType* predicts,
    IdType* output_token_ids,
    IdType* output_accepted_token_num,
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    DType* draft_probs,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single = 1,
    DType threshold_acc = 1,
    bool deterministic = true,
    hipStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd<uint32_t>(16 / sizeof(DType), d);
  const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  float capped_threshold_acc = fmaxf(threshold_acc, 1e-9f);

  DISPATCH_ALIGNED_VEC_SIZE(
      vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
        auto kernel_fn = TreeSpeculativeSamplingTargetOnly<
            BLOCK_THREADS,
            SCAN_ALGO,
            REDUCE_ALGO,
            VEC_SIZE,
            DETERMINISTIC,
            DType,
            IdType,
            IdType2>;

        hipError_t err = hipFuncSetAttribute(
            reinterpret_cast<const void*>(kernel_fn), hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (err != hipSuccess) return err;

        hipLaunchKernelGGL(
            kernel_fn,
            nblks,
            nthrs,
            smem_size,
            stream,
            predicts,
            output_token_ids,
            output_accepted_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            batch_size,
            num_speculative_tokens,
            num_draft_tokens,
            d,
            threshold_single,
            capped_threshold_acc);
      })});
  return hipSuccess;
}

}  // namespace speculative_hip

#endif  // SPECULATIVE_SAMPLING_HIP_CUH_
