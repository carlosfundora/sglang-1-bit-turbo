"""RDNA2 (gfx1030) HIP paged flash-decode attention as a torch op.

Device-resident, paged-KV, fp16 analog of the canonical
`rs_rdna2_kernels::flash_decode` split-K kernel (projects/rust + build/kernels/
flash-decode-hip). Operates IN-PLACE on the on-GPU KV cache via kv_indptr/kv_indices
(no host round-trips, no copies) so it can replace sglang's Triton decode op in the
serving loop. Adaptive split-K: gang W in {1,2,4} Wave32 warps per (request, q-head)
and split the gathered key range across them, merging the partial online-softmaxes via
LDS — the achievable "wider wavefront" on gfx1030 (ROCm won't emit real wave64) and the
parallelism that fills the GPU on small-batch decode.

Contract (matches sglang triton decode_attention_fwd inputs):
  q          [N, Hq, D]   fp16   (N decode tokens, one query each)
  k_buffer   [T, Hk, D]   fp16   (paged KV cache, gathered by kv_indices)
  v_buffer   [T, Hk, D]   fp16
  o          [N, Hq, D]   fp16   (written)
  kv_indptr  [N+1]        int32  (per-request key-range offsets into kv_indices)
  kv_indices [total]      int32  (token slots into k_buffer/v_buffer)
  sm_scale   float        (softmax scale, typically 1/sqrt(D))
GQA: kv_head = q_head / (Hq/Hk). No causal mask (kv_indices is the valid set).
Constraints: D % 32 == 0 and D <= 256.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_DECL = """
void rdna2_hip_paged_decode(
    torch::Tensor& o, torch::Tensor& q, torch::Tensor& k_buffer, torch::Tensor& v_buffer,
    torch::Tensor& kv_indptr, torch::Tensor& kv_indices, double sm_scale);
"""

_SRC = r"""
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <float.h>

namespace rdna2hd {
constexpr int WARP_SIZE = 32;
constexpr int MAX_ELEMS = 8;   // D/32 <= 8 -> D <= 256
constexpr int MAX_W = 4;

template <int W>
__global__ __attribute__((amdgpu_flat_work_group_size(W * 32, W * 32)))
void paged_decode_kernel(
    __half* __restrict__ o,            // [N, Hq, D]
    const __half* __restrict__ q,      // [N, Hq, D]
    const __half* __restrict__ k_buf,  // [T, Hk, D]
    const __half* __restrict__ v_buf,  // [T, Hk, D]
    const int* __restrict__ kv_indptr, // [N+1]
    const int* __restrict__ kv_indices,// [total]
    int Hq, int Hk, int D, float scale)
{
    const int row = blockIdx.x;        // (req * Hq + hq)
    const int req = row / Hq;
    const int hq  = row % Hq;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int elems = D / WARP_SIZE;
    const int hkv = hq / (Hq / Hk);

    const int kstart = kv_indptr[req];
    const int kend   = kv_indptr[req + 1];

    const long q_base = ((long)row) * D;
    float qreg[MAX_ELEMS], acc[MAX_ELEMS];
    #pragma unroll
    for (int i = 0; i < MAX_ELEMS; i++) { qreg[i] = 0.f; acc[i] = 0.f; }
    for (int i = 0; i < elems; i++) qreg[i] = __half2float(q[q_base + lane * elems + i]);

    float m = -FLT_MAX, l = 0.f;
    for (int s = kstart + warp; s < kend; s += W) {
        const int tok = kv_indices[s];
        const long kv_base = ((long)tok * Hk + hkv) * D;
        float dot = 0.f;
        for (int i = 0; i < elems; i++)
            dot += qreg[i] * __half2float(k_buf[kv_base + lane * elems + i]);
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) dot += __shfl_xor(dot, off);
        const float score = dot * scale;
        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float p = __expf(score - m_new);
        l = l * corr + p;
        for (int i = 0; i < elems; i++)
            acc[i] = acc[i] * corr + p * __half2float(v_buf[kv_base + lane * elems + i]);
        m = m_new;
    }

    if (W == 1) {
        const float inv = (l > 0.f) ? 1.f / l : 0.f;
        for (int i = 0; i < elems; i++) o[q_base + lane * elems + i] = __float2half(acc[i] * inv);
        return;
    }
    __shared__ float sm[MAX_W];
    __shared__ float sl[MAX_W];
    __shared__ float sacc[MAX_W][WARP_SIZE * MAX_ELEMS];
    sm[warp] = m; sl[warp] = l;
    for (int i = 0; i < elems; i++) sacc[warp][lane * elems + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float mg = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < W; w++) mg = fmaxf(mg, sm[w]);
        float lg = 0.f, f[W];
        #pragma unroll
        for (int w = 0; w < W; w++) { f[w] = __expf(sm[w] - mg); lg += sl[w] * f[w]; }
        const float inv = (lg > 0.f) ? 1.f / lg : 0.f;
        for (int i = 0; i < elems; i++) {
            float ov = 0.f;
            #pragma unroll
            for (int w = 0; w < W; w++) ov += sacc[w][lane * elems + i] * f[w];
            o[q_base + lane * elems + i] = __float2half(ov * inv);
        }
    }
}
}  // namespace rdna2hd

void rdna2_hip_paged_decode(
    torch::Tensor& o, torch::Tensor& q, torch::Tensor& k_buffer, torch::Tensor& v_buffer,
    torch::Tensor& kv_indptr, torch::Tensor& kv_indices, double sm_scale)
{
    const int N  = q.size(0);
    const int Hq = q.size(1);
    const int D  = q.size(2);
    const int Hk = k_buffer.size(1);
    TORCH_CHECK(D % rdna2hd::WARP_SIZE == 0 && D <= 32 * rdna2hd::MAX_ELEMS, "D must be %32 and <=256");
    TORCH_CHECK(Hq % Hk == 0, "Hq must be a multiple of Hk");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k_buffer.scalar_type() == at::kHalf,
                "rdna2_hip_paged_decode expects fp16 q/k/v");

    // Adaptive split-K: aim to fill the GPU; cap by MAX_W; require >=8 keys/warp.
    int rows = N * Hq, cu = 20;
    {
        hipDeviceProp_t p;
        if (hipGetDeviceProperties(&p, 0) == hipSuccess && p.multiProcessorCount > 0) cu = p.multiProcessorCount;
    }
    // conservative: use the smallest per-request key count is unknown here; scale by rows only.
    int W = 1;
    while (W < rdna2hd::MAX_W && rows * (W * 2) <= cu * 32) W *= 2;

    dim3 grid(rows);
    dim3 block(rdna2hd::WARP_SIZE * W);
    auto stream = at::hip::getCurrentHIPStream();
    auto* op = reinterpret_cast<__half*>(o.data_ptr<at::Half>());
    auto* qp = reinterpret_cast<__half*>(q.data_ptr<at::Half>());
    auto* kp = reinterpret_cast<__half*>(k_buffer.data_ptr<at::Half>());
    auto* vp = reinterpret_cast<__half*>(v_buffer.data_ptr<at::Half>());
    auto* ip = kv_indptr.data_ptr<int>();
    auto* ix = kv_indices.data_ptr<int>();
    float scale = (float)sm_scale;
    #define LAUNCH(WW) rdna2hd::paged_decode_kernel<WW><<<grid, block, 0, stream>>>(op,qp,kp,vp,ip,ix,Hq,Hk,D,scale)
    switch (W) { case 4: LAUNCH(4); break; case 2: LAUNCH(2); break; default: LAUNCH(1); break; }
    #undef LAUNCH
}
"""

_mod = None


def _get_module():
    global _mod
    if _mod is not None:
        return _mod
    try:
        from torch.utils.cpp_extension import load_inline

        _mod = load_inline(
            name="rdna2_hip_paged_decode",
            cpp_sources=_DECL,
            cuda_sources=_SRC,
            functions=["rdna2_hip_paged_decode"],
            extra_cuda_cflags=["--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64"],
            verbose=False,
        )
        logger.info("RDNA2 HIP paged flash-decode: compiled via torch cpp_extension")
    except Exception as e:  # pragma: no cover
        logger.warning(f"RDNA2 HIP paged flash-decode compile failed: {e}")
        _mod = None
    return _mod


def is_available() -> bool:
    return _get_module() is not None


def paged_decode(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, sm_scale):
    """q [N,Hq,D] fp16, paged KV via kv_indptr/kv_indices -> writes o [N,Hq,D] fp16."""
    mod = _get_module()
    if mod is None:
        raise RuntimeError("rdna2_hip_paged_decode unavailable")
    mod.rdna2_hip_paged_decode(
        o.contiguous(),
        q.contiguous(),
        k_buffer,
        v_buffer,
        kv_indptr.to(torch.int32),
        kv_indices.to(torch.int32),
        float(sm_scale),
    )
    return o
